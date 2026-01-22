import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
class _SysProcessEvent(_ProcessEvent):
    """
    There is three kind of processing according to each event:

      1. special handling (deletion from internal container, bug, ...).
      2. default treatment: which is applied to the majority of events.
      3. IN_ISDIR is never sent alone, he is piggybacked with a standard
         event, he is not processed as the others events, instead, its
         value is captured and appropriately aggregated to dst event.
    """

    def __init__(self, wm, notifier):
        """

        @param wm: Watch Manager.
        @type wm: WatchManager instance
        @param notifier: Notifier.
        @type notifier: Notifier instance
        """
        self._watch_manager = wm
        self._notifier = notifier
        self._mv_cookie = {}
        self._mv = {}

    def cleanup(self):
        """
        Cleanup (delete) old (>1mn) records contained in self._mv_cookie
        and self._mv.
        """
        date_cur_ = datetime.now()
        for seq in (self._mv_cookie, self._mv):
            for k in list(seq.keys()):
                if date_cur_ - seq[k][1] > timedelta(minutes=1):
                    log.debug('Cleanup: deleting entry %s', seq[k][0])
                    del seq[k]

    def process_IN_CREATE(self, raw_event):
        """
        If the event affects a directory and the auto_add flag of the
        targetted watch is set to True, a new watch is added on this
        new directory, with the same attribute values than those of
        this watch.
        """
        if raw_event.mask & IN_ISDIR:
            watch_ = self._watch_manager.get_watch(raw_event.wd)
            created_dir = os.path.join(watch_.path, raw_event.name)
            if watch_.auto_add and (not watch_.exclude_filter(created_dir)):
                addw = self._watch_manager.add_watch
                addw_ret = addw(created_dir, watch_.mask, proc_fun=watch_.proc_fun, rec=False, auto_add=watch_.auto_add, exclude_filter=watch_.exclude_filter)
                created_dir_wd = addw_ret.get(created_dir)
                if created_dir_wd is not None and created_dir_wd > 0 and os.path.isdir(created_dir):
                    try:
                        for name in os.listdir(created_dir):
                            inner = os.path.join(created_dir, name)
                            if self._watch_manager.get_wd(inner) is not None:
                                continue
                            if os.path.isfile(inner):
                                flags = IN_CREATE
                            elif os.path.isdir(inner):
                                flags = IN_CREATE | IN_ISDIR
                            else:
                                continue
                            rawevent = _RawEvent(created_dir_wd, flags, 0, name)
                            self._notifier.append_event(rawevent)
                    except OSError as err:
                        msg = 'process_IN_CREATE, invalid directory: %s'
                        log.debug(msg % str(err))
        return self.process_default(raw_event)

    def process_IN_MOVED_FROM(self, raw_event):
        """
        Map the cookie with the source path (+ date for cleaning).
        """
        watch_ = self._watch_manager.get_watch(raw_event.wd)
        path_ = watch_.path
        src_path = os.path.normpath(os.path.join(path_, raw_event.name))
        self._mv_cookie[raw_event.cookie] = (src_path, datetime.now())
        return self.process_default(raw_event, {'cookie': raw_event.cookie})

    def process_IN_MOVED_TO(self, raw_event):
        """
        Map the source path with the destination path (+ date for
        cleaning).
        """
        watch_ = self._watch_manager.get_watch(raw_event.wd)
        path_ = watch_.path
        dst_path = os.path.normpath(os.path.join(path_, raw_event.name))
        mv_ = self._mv_cookie.get(raw_event.cookie)
        to_append = {'cookie': raw_event.cookie}
        if mv_ is not None:
            self._mv[mv_[0]] = (dst_path, datetime.now())
            to_append['src_pathname'] = mv_[0]
        elif raw_event.mask & IN_ISDIR and watch_.auto_add and (not watch_.exclude_filter(dst_path)):
            self._watch_manager.add_watch(dst_path, watch_.mask, proc_fun=watch_.proc_fun, rec=True, auto_add=True, exclude_filter=watch_.exclude_filter)
        return self.process_default(raw_event, to_append)

    def process_IN_MOVE_SELF(self, raw_event):
        """
        STATUS: the following bug has been fixed in recent kernels (FIXME:
        which version ?). Now it raises IN_DELETE_SELF instead.

        Old kernels were bugged, this event raised when the watched item
        were moved, so we had to update its path, but under some circumstances
        it was impossible: if its parent directory and its destination
        directory wasn't watched. The kernel (see include/linux/fsnotify.h)
        doesn't bring us enough informations like the destination path of
        moved items.
        """
        watch_ = self._watch_manager.get_watch(raw_event.wd)
        src_path = watch_.path
        mv_ = self._mv.get(src_path)
        if mv_:
            dest_path = mv_[0]
            watch_.path = dest_path
            src_path += os.path.sep
            src_path_len = len(src_path)
            for w in self._watch_manager.watches.values():
                if w.path.startswith(src_path):
                    w.path = os.path.join(dest_path, w.path[src_path_len:])
        else:
            log.error("The pathname '%s' of this watch %s has probably changed and couldn't be updated, so it cannot be trusted anymore. To fix this error move directories/files only between watched parents directories, in this case e.g. put a watch on '%s'.", watch_.path, watch_, os.path.normpath(os.path.join(watch_.path, os.path.pardir)))
            if not watch_.path.endswith('-unknown-path'):
                watch_.path += '-unknown-path'
        return self.process_default(raw_event)

    def process_IN_Q_OVERFLOW(self, raw_event):
        """
        Only signal an overflow, most of the common flags are irrelevant
        for this event (path, wd, name).
        """
        return Event({'mask': raw_event.mask})

    def process_IN_IGNORED(self, raw_event):
        """
        The watch descriptor raised by this event is now ignored (forever),
        it can be safely deleted from the watch manager dictionary.
        After this event we can be sure that neither the event queue nor
        the system will raise an event associated to this wd again.
        """
        event_ = self.process_default(raw_event)
        self._watch_manager.del_watch(raw_event.wd)
        return event_

    def process_default(self, raw_event, to_append=None):
        """
        Commons handling for the followings events:

        IN_ACCESS, IN_MODIFY, IN_ATTRIB, IN_CLOSE_WRITE, IN_CLOSE_NOWRITE,
        IN_OPEN, IN_DELETE, IN_DELETE_SELF, IN_UNMOUNT.
        """
        watch_ = self._watch_manager.get_watch(raw_event.wd)
        if raw_event.mask & (IN_DELETE_SELF | IN_MOVE_SELF):
            dir_ = watch_.dir
        else:
            dir_ = bool(raw_event.mask & IN_ISDIR)
        dict_ = {'wd': raw_event.wd, 'mask': raw_event.mask, 'path': watch_.path, 'name': raw_event.name, 'dir': dir_}
        if COMPATIBILITY_MODE:
            dict_['is_dir'] = dir_
        if to_append is not None:
            dict_.update(to_append)
        return Event(dict_)