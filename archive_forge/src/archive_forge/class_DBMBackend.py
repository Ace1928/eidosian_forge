from contextlib import contextmanager
import dbm
import os
import threading
from ..api import BytesBackend
from ..api import NO_VALUE
from ... import util
class DBMBackend(BytesBackend):
    """A file-backend using a dbm file to store keys.

    Basic usage::

        from dogpile.cache import make_region

        region = make_region().configure(
            'dogpile.cache.dbm',
            expiration_time = 3600,
            arguments = {
                "filename":"/path/to/cachefile.dbm"
            }
        )

    DBM access is provided using the Python ``anydbm`` module,
    which selects a platform-specific dbm module to use.
    This may be made to be more configurable in a future
    release.

    Note that different dbm modules have different behaviors.
    Some dbm implementations handle their own locking, while
    others don't.  The :class:`.DBMBackend` uses a read/write
    lockfile by default, which is compatible even with those
    DBM implementations for which this is unnecessary,
    though the behavior can be disabled.

    The DBM backend by default makes use of two lockfiles.
    One is in order to protect the DBM file itself from
    concurrent writes, the other is to coordinate
    value creation (i.e. the dogpile lock).  By default,
    these lockfiles use the ``flock()`` system call
    for locking; this is **only available on Unix
    platforms**.   An alternative lock implementation, such as one
    which is based on threads or uses a third-party system
    such as `portalocker <https://pypi.python.org/pypi/portalocker>`_,
    can be dropped in using the ``lock_factory`` argument
    in conjunction with the :class:`.AbstractFileLock` base class.

    Currently, the dogpile lock is against the entire
    DBM file, not per key.   This means there can
    only be one "creator" job running at a time
    per dbm file.

    A future improvement might be to have the dogpile lock
    using a filename that's based on a modulus of the key.
    Locking on a filename that uniquely corresponds to the
    key is problematic, since it's not generally safe to
    delete lockfiles as the application runs, implying an
    unlimited number of key-based files would need to be
    created and never deleted.

    Parameters to the ``arguments`` dictionary are
    below.

    :param filename: path of the filename in which to
     create the DBM file.  Note that some dbm backends
     will change this name to have additional suffixes.
    :param rw_lockfile: the name of the file to use for
     read/write locking.  If omitted, a default name
     is used by appending the suffix ".rw.lock" to the
     DBM filename.  If False, then no lock is used.
    :param dogpile_lockfile: the name of the file to use
     for value creation, i.e. the dogpile lock.  If
     omitted, a default name is used by appending the
     suffix ".dogpile.lock" to the DBM filename. If
     False, then dogpile.cache uses the default dogpile
     lock, a plain thread-based mutex.
    :param lock_factory: a function or class which provides
     for a read/write lock.  Defaults to :class:`.FileLock`.
     Custom implementations need to implement context-manager
     based ``read()`` and ``write()`` functions - the
     :class:`.AbstractFileLock` class is provided as a base class
     which provides these methods based on individual read/write lock
     functions.  E.g. to replace the lock with the dogpile.core
     :class:`.ReadWriteMutex`::

        from dogpile.core.readwrite_lock import ReadWriteMutex
        from dogpile.cache.backends.file import AbstractFileLock

        class MutexLock(AbstractFileLock):
            def __init__(self, filename):
                self.mutex = ReadWriteMutex()

            def acquire_read_lock(self, wait):
                ret = self.mutex.acquire_read_lock(wait)
                return wait or ret

            def acquire_write_lock(self, wait):
                ret = self.mutex.acquire_write_lock(wait)
                return wait or ret

            def release_read_lock(self):
                return self.mutex.release_read_lock()

            def release_write_lock(self):
                return self.mutex.release_write_lock()

        from dogpile.cache import make_region

        region = make_region().configure(
            "dogpile.cache.dbm",
            expiration_time=300,
            arguments={
                "filename": "file.dbm",
                "lock_factory": MutexLock
            }
        )

     While the included :class:`.FileLock` uses ``os.flock()``, a
     windows-compatible implementation can be built using a library
     such as `portalocker <https://pypi.python.org/pypi/portalocker>`_.

     .. versionadded:: 0.5.2



    """

    def __init__(self, arguments):
        self.filename = os.path.abspath(os.path.normpath(arguments['filename']))
        dir_, filename = os.path.split(self.filename)
        self.lock_factory = arguments.get('lock_factory', FileLock)
        self._rw_lock = self._init_lock(arguments.get('rw_lockfile'), '.rw.lock', dir_, filename)
        self._dogpile_lock = self._init_lock(arguments.get('dogpile_lockfile'), '.dogpile.lock', dir_, filename, util.KeyReentrantMutex.factory)
        self._init_dbm_file()

    def _init_lock(self, argument, suffix, basedir, basefile, wrapper=None):
        if argument is None:
            lock = self.lock_factory(os.path.join(basedir, basefile + suffix))
        elif argument is not False:
            lock = self.lock_factory(os.path.abspath(os.path.normpath(argument)))
        else:
            return None
        if wrapper:
            lock = wrapper(lock)
        return lock

    def _init_dbm_file(self):
        exists = os.access(self.filename, os.F_OK)
        if not exists:
            for ext in ('db', 'dat', 'pag', 'dir'):
                if os.access(self.filename + os.extsep + ext, os.F_OK):
                    exists = True
                    break
        if not exists:
            fh = dbm.open(self.filename, 'c')
            fh.close()

    def get_mutex(self, key):
        if self._dogpile_lock:
            return self._dogpile_lock(key)
        else:
            return None

    @contextmanager
    def _use_rw_lock(self, write):
        if self._rw_lock is None:
            yield
        elif write:
            with self._rw_lock.write():
                yield
        else:
            with self._rw_lock.read():
                yield

    @contextmanager
    def _dbm_file(self, write):
        with self._use_rw_lock(write):
            with dbm.open(self.filename, 'w' if write else 'r') as dbm_obj:
                yield dbm_obj

    def get_serialized(self, key):
        with self._dbm_file(False) as dbm_obj:
            if hasattr(dbm_obj, 'get'):
                value = dbm_obj.get(key, NO_VALUE)
            else:
                try:
                    value = dbm_obj[key]
                except KeyError:
                    value = NO_VALUE
            return value

    def get_serialized_multi(self, keys):
        return [self.get_serialized(key) for key in keys]

    def set_serialized(self, key, value):
        with self._dbm_file(True) as dbm_obj:
            dbm_obj[key] = value

    def set_serialized_multi(self, mapping):
        with self._dbm_file(True) as dbm_obj:
            for key, value in mapping.items():
                dbm_obj[key] = value

    def delete(self, key):
        with self._dbm_file(True) as dbm_obj:
            try:
                del dbm_obj[key]
            except KeyError:
                pass

    def delete_multi(self, keys):
        with self._dbm_file(True) as dbm_obj:
            for key in keys:
                try:
                    del dbm_obj[key]
                except KeyError:
                    pass