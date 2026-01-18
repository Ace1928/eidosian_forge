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
def update_watch(self, wd, mask=None, proc_fun=None, rec=False, auto_add=False, quiet=True):
    """
        Update existing watch descriptors |wd|. The |mask| value, the
        processing object |proc_fun|, the recursive param |rec| and the
        |auto_add| and |quiet| flags can all be updated.

        @param wd: Watch Descriptor to update. Also accepts a list of
                   watch descriptors.
        @type wd: int or list of int
        @param mask: Optional new bitmask of events.
        @type mask: int
        @param proc_fun: Optional new processing function.
        @type proc_fun: function or ProcessEvent instance or instance of
                        one of its subclasses or callable object.
        @param rec: Optionally adds watches recursively on all
                    subdirectories contained into |wd| directory.
        @type rec: bool
        @param auto_add: Automatically adds watches on newly created
                         directories in the watch's path corresponding to |wd|.
                         If |auto_add| is True, IN_CREATE is ored with |mask|
                         when the watch is updated.
        @type auto_add: bool
        @param quiet: If False raises a WatchManagerError exception on
                      error. See example not_quiet.py
        @type quiet: bool
        @return: dict of watch descriptors associated to booleans values.
                 True if the corresponding wd has been successfully
                 updated, False otherwise.
        @rtype: dict of {int: bool}
        """
    lwd = self.__format_param(wd)
    if rec:
        lwd = self.__get_sub_rec(lwd)
    ret_ = {}
    for awd in lwd:
        apath = self.get_path(awd)
        if not apath or awd < 0:
            err = 'update_watch: invalid WD=%d' % awd
            if quiet:
                log.error(err)
                continue
            raise WatchManagerError(err, ret_)
        if mask:
            wd_ = self._inotify_wrapper.inotify_add_watch(self._fd, apath, mask)
            if wd_ < 0:
                ret_[awd] = False
                err = 'update_watch: cannot update %s WD=%d, %s' % (apath, wd_, self._inotify_wrapper.str_errno())
                if quiet:
                    log.error(err)
                    continue
                raise WatchManagerError(err, ret_)
            assert awd == wd_
        if proc_fun or auto_add:
            watch_ = self._wmd[awd]
        if proc_fun:
            watch_.proc_fun = proc_fun
        if auto_add:
            watch_.auto_add = auto_add
        ret_[awd] = True
        log.debug('Updated watch - %s', self._wmd[awd])
    return ret_