import errno
import os
from time import time as _uniquefloat
from twisted.python.runtime import platform
from os import rename
class FilesystemLock:
    """
    A mutex.

    This relies on the filesystem property that creating
    a symlink is an atomic operation and that it will
    fail if the symlink already exists.  Deleting the
    symlink will release the lock.

    @ivar name: The name of the file associated with this lock.

    @ivar clean: Indicates whether this lock was released cleanly by its
        last owner.  Only meaningful after C{lock} has been called and
        returns True.

    @ivar locked: Indicates whether the lock is currently held by this
        object.
    """
    clean = None
    locked = False

    def __init__(self, name):
        self.name = name

    def lock(self):
        """
        Acquire this lock.

        @rtype: C{bool}
        @return: True if the lock is acquired, false otherwise.

        @raise OSError: Any exception L{os.symlink()} may raise,
            other than L{errno.EEXIST}.
        """
        clean = True
        while True:
            try:
                symlink(str(os.getpid()), self.name)
            except OSError as e:
                if _windows and e.errno in (errno.EACCES, errno.EIO):
                    return False
                if e.errno == errno.EEXIST:
                    try:
                        pid = readlink(self.name)
                    except OSError as e:
                        if e.errno == errno.ENOENT:
                            continue
                        elif _windows and e.errno == errno.EACCES:
                            return False
                        raise
                    try:
                        if kill is not None:
                            kill(int(pid), 0)
                    except OSError as e:
                        if e.errno == errno.ESRCH:
                            try:
                                rmlink(self.name)
                            except OSError as e:
                                if e.errno == errno.ENOENT:
                                    continue
                                raise
                            clean = False
                            continue
                        raise
                    return False
                raise
            self.locked = True
            self.clean = clean
            return True

    def unlock(self):
        """
        Release this lock.

        This deletes the directory with the given name.

        @raise OSError: Any exception L{os.readlink()} may raise.
        @raise ValueError: If the lock is not owned by this process.
        """
        pid = readlink(self.name)
        if int(pid) != os.getpid():
            raise ValueError(f'Lock {self.name!r} not owned by this process')
        rmlink(self.name)
        self.locked = False