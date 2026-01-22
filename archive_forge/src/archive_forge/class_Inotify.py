from __future__ import with_statement
import os
import errno
import struct
import threading
import ctypes
import ctypes.util
from functools import reduce
from ctypes import c_int, c_char_p, c_uint32
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import UnsupportedLibc
class Inotify(object):
    """
    Linux inotify(7) API wrapper class.

    :param path:
        The directory path for which we want an inotify object.
    :type path:
        :class:`bytes`
    :param recursive:
        ``True`` if subdirectories should be monitored; ``False`` otherwise.
    """

    def __init__(self, path, recursive=False, event_mask=WATCHDOG_ALL_EVENTS):
        inotify_fd = inotify_init()
        if inotify_fd == -1:
            Inotify._raise_error()
        self._inotify_fd = inotify_fd
        self._lock = threading.Lock()
        self._wd_for_path = dict()
        self._path_for_wd = dict()
        self._path = path
        self._event_mask = event_mask
        self._is_recursive = recursive
        self._add_dir_watch(path, recursive, event_mask)
        self._moved_from_events = dict()

    @property
    def event_mask(self):
        """The event mask for this inotify instance."""
        return self._event_mask

    @property
    def path(self):
        """The path associated with the inotify instance."""
        return self._path

    @property
    def is_recursive(self):
        """Whether we are watching directories recursively."""
        return self._is_recursive

    @property
    def fd(self):
        """The file descriptor associated with the inotify instance."""
        return self._inotify_fd

    def clear_move_records(self):
        """Clear cached records of MOVED_FROM events"""
        self._moved_from_events = dict()

    def source_for_move(self, destination_event):
        """
        The source path corresponding to the given MOVED_TO event.

        If the source path is outside the monitored directories, None
        is returned instead.
        """
        if destination_event.cookie in self._moved_from_events:
            return self._moved_from_events[destination_event.cookie].src_path
        else:
            return None

    def remember_move_from_event(self, event):
        """
        Save this event as the source event for future MOVED_TO events to
        reference.
        """
        self._moved_from_events[event.cookie] = event

    def add_watch(self, path):
        """
        Adds a watch for the given path.

        :param path:
            Path to begin monitoring.
        """
        with self._lock:
            self._add_watch(path, self._event_mask)

    def remove_watch(self, path):
        """
        Removes a watch for the given path.

        :param path:
            Path string for which the watch will be removed.
        """
        with self._lock:
            wd = self._wd_for_path.pop(path)
            del self._path_for_wd[wd]
            if inotify_rm_watch(self._inotify_fd, wd) == -1:
                Inotify._raise_error()

    def close(self):
        """
        Closes the inotify instance and removes all associated watches.
        """
        with self._lock:
            if self._path in self._wd_for_path:
                wd = self._wd_for_path[self._path]
                inotify_rm_watch(self._inotify_fd, wd)
            os.close(self._inotify_fd)

    def read_events(self, event_buffer_size=DEFAULT_EVENT_BUFFER_SIZE):
        """
        Reads events from inotify and yields them.
        """

        def _recursive_simulate(src_path):
            events = []
            for root, dirnames, filenames in os.walk(src_path):
                for dirname in dirnames:
                    try:
                        full_path = os.path.join(root, dirname)
                        wd_dir = self._add_watch(full_path, self._event_mask)
                        e = InotifyEvent(wd_dir, InotifyConstants.IN_CREATE | InotifyConstants.IN_ISDIR, 0, dirname, full_path)
                        events.append(e)
                    except OSError:
                        pass
                for filename in filenames:
                    full_path = os.path.join(root, filename)
                    wd_parent_dir = self._wd_for_path[os.path.dirname(full_path)]
                    e = InotifyEvent(wd_parent_dir, InotifyConstants.IN_CREATE, 0, filename, full_path)
                    events.append(e)
            return events
        event_buffer = None
        while True:
            try:
                event_buffer = os.read(self._inotify_fd, event_buffer_size)
            except OSError as e:
                if e.errno == errno.EINTR:
                    continue
            break
        with self._lock:
            event_list = []
            for wd, mask, cookie, name in Inotify._parse_event_buffer(event_buffer):
                if wd == -1:
                    continue
                wd_path = self._path_for_wd[wd]
                src_path = os.path.join(wd_path, name) if name else wd_path
                inotify_event = InotifyEvent(wd, mask, cookie, name, src_path)
                if inotify_event.is_moved_from:
                    self.remember_move_from_event(inotify_event)
                elif inotify_event.is_moved_to:
                    move_src_path = self.source_for_move(inotify_event)
                    if move_src_path in self._wd_for_path:
                        moved_wd = self._wd_for_path[move_src_path]
                        del self._wd_for_path[move_src_path]
                        self._wd_for_path[inotify_event.src_path] = moved_wd
                        self._path_for_wd[moved_wd] = inotify_event.src_path
                    src_path = os.path.join(wd_path, name)
                    inotify_event = InotifyEvent(wd, mask, cookie, name, src_path)
                if inotify_event.is_ignored:
                    path = self._path_for_wd.pop(wd)
                    if self._wd_for_path[path] == wd:
                        del self._wd_for_path[path]
                    continue
                event_list.append(inotify_event)
                if self.is_recursive and inotify_event.is_directory and inotify_event.is_create:
                    try:
                        self._add_watch(src_path, self._event_mask)
                    except OSError:
                        continue
                    event_list.extend(_recursive_simulate(src_path))
        return event_list

    def _add_dir_watch(self, path, recursive, mask):
        """
        Adds a watch (optionally recursively) for the given directory path
        to monitor events specified by the mask.

        :param path:
            Path to monitor
        :param recursive:
            ``True`` to monitor recursively.
        :param mask:
            Event bit mask.
        """
        if not os.path.isdir(path):
            raise OSError('Path is not a directory')
        self._add_watch(path, mask)
        if recursive:
            for root, dirnames, _ in os.walk(path):
                for dirname in dirnames:
                    full_path = os.path.join(root, dirname)
                    if os.path.islink(full_path):
                        continue
                    self._add_watch(full_path, mask)

    def _add_watch(self, path, mask):
        """
        Adds a watch for the given path to monitor events specified by the
        mask.

        :param path:
            Path to monitor
        :param mask:
            Event bit mask.
        """
        wd = inotify_add_watch(self._inotify_fd, path, mask)
        if wd == -1:
            Inotify._raise_error()
        self._wd_for_path[path] = wd
        self._path_for_wd[wd] = path
        return wd

    @staticmethod
    def _raise_error():
        """
        Raises errors for inotify failures.
        """
        err = ctypes.get_errno()
        if err == errno.ENOSPC:
            raise OSError('inotify watch limit reached')
        elif err == errno.EMFILE:
            raise OSError('inotify instance limit reached')
        else:
            raise OSError(os.strerror(err))

    @staticmethod
    def _parse_event_buffer(event_buffer):
        """
        Parses an event buffer of ``inotify_event`` structs returned by
        inotify::

            struct inotify_event {
                __s32 wd;            /* watch descriptor */
                __u32 mask;          /* watch mask */
                __u32 cookie;        /* cookie to synchronize two events */
                __u32 len;           /* length (including nulls) of name */
                char  name[0];       /* stub for possible name */
            };

        The ``cookie`` member of this struct is used to pair two related
        events, for example, it pairs an IN_MOVED_FROM event with an
        IN_MOVED_TO event.
        """
        i = 0
        while i + 16 <= len(event_buffer):
            wd, mask, cookie, length = struct.unpack_from('iIII', event_buffer, i)
            name = event_buffer[i + 16:i + 16 + length].rstrip(b'\x00')
            i += 16 + length
            yield (wd, mask, cookie, name)