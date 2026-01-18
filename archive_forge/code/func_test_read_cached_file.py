import os
from unittest import mock
import fixtures
import oslo_config
from oslotest import base as test_base
from oslo_policy import _cache_handler as _ch
def test_read_cached_file(self):
    file_cache = {}
    path = os.path.join(self.tmpdir.path, 'tmpfile')
    with open(path, 'w+') as fp:
        fp.write('test')
    reloaded, data = _ch.read_cached_file(file_cache, path)
    self.assertEqual('test', data)
    self.assertTrue(reloaded)
    reloaded, data = _ch.read_cached_file(file_cache, path)
    self.assertEqual('test', data)
    self.assertFalse(reloaded)
    reloaded, data = _ch.read_cached_file(file_cache, path, force_reload=True)
    self.assertEqual('test', data)
    self.assertTrue(reloaded)