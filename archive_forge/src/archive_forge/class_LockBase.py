from __future__ import absolute_import
import functools
import os
import socket
import threading
import warnings
class LockBase(_SharedBase):
    """Base class for platform-specific lock classes."""

    def __init__(self, path, threaded=True, timeout=None):
        """
        >>> lock = LockBase('somefile')
        >>> lock = LockBase('somefile', threaded=False)
        """
        super(LockBase, self).__init__(path)
        self.lock_file = os.path.abspath(path) + '.lock'
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        if threaded:
            t = threading.current_thread()
            ident = getattr(t, 'ident', hash(t))
            self.tname = '-%x' % (ident & 4294967295)
        else:
            self.tname = ''
        dirname = os.path.dirname(self.lock_file)
        self.unique_name = os.path.join(dirname, '%s%s.%s%s' % (self.hostname, self.tname, self.pid, hash(self.path)))
        self.timeout = timeout

    def is_locked(self):
        """
        Tell whether or not the file is locked.
        """
        raise NotImplemented('implement in subclass')

    def i_am_locking(self):
        """
        Return True if this object is locking the file.
        """
        raise NotImplemented('implement in subclass')

    def break_lock(self):
        """
        Remove a lock.  Useful if a locking thread failed to unlock.
        """
        raise NotImplemented('implement in subclass')

    def __repr__(self):
        return '<%s: %r -- %r>' % (self.__class__.__name__, self.unique_name, self.path)