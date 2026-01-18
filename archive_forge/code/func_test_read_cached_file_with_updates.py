import os
from unittest import mock
import fixtures
import oslo_config
from oslotest import base as test_base
from oslo_policy import _cache_handler as _ch
def test_read_cached_file_with_updates(self):
    file_cache = {}
    path = os.path.join(self.tmpdir.path, 'tmpfile')
    with open(path, 'w+') as fp:
        fp.write('test')
    reloaded, data = _ch.read_cached_file(file_cache, path)
    times = (os.stat(path).st_atime + 1, os.stat(path).st_mtime + 1)
    os.utime(path, times)
    reloaded, data = _ch.read_cached_file(file_cache, path)
    self.assertTrue(reloaded)