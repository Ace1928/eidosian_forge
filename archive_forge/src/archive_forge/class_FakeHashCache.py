import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
class FakeHashCache(HashCache):
    """Hashcache that consults a fake clock rather than the real one.

    This lets us examine how old or new files would be handled, without
    actually having to wait for time to pass.
    """

    def __init__(self):
        HashCache.__init__(self, '.', 'hashcache')
        self._files = {}
        self._clock = 0

    def put_file(self, filename, file_contents):
        abspath = './' + filename
        self._files[abspath] = (file_contents, self._clock)

    def _fingerprint(self, abspath, fs=None):
        entry = self._files[abspath]
        return (len(entry[0]), entry[1], entry[1], 10, 20, stat.S_IFREG | 384)

    def _really_sha1_file(self, abspath, filters):
        if abspath in self._files:
            return sha1(self._files[abspath][0])
        else:
            return None

    def _cutoff_time(self):
        return self._clock - 2

    def pretend_to_sleep(self, secs):
        self._clock += secs