import logging
import os
class SimpleLockFile:
    _fp = None

    def __init__(self, path):
        self._path = path
        try:
            fp = open(path, 'r+')
        except OSError:
            fp = open(path, 'a+')
        try:
            _lock_file(fp)
            self._fp = fp
        except BaseException:
            fp.close()
            raise
        self._on_lock()
        fp.flush()

    def close(self):
        if self._fp is not None:
            _unlock_file(self._fp)
            self._fp.close()
            self._fp = None

    def _on_lock(self):
        """
        Allow subclasses to supply behavior to occur following
        lock acquisition.
        """