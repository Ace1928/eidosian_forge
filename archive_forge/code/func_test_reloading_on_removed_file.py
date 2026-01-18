import os
from unittest import mock
import fixtures
import oslo_config
from oslotest import base as test_base
from oslo_policy import _cache_handler as _ch
@mock.patch.object(_ch, 'LOG')
def test_reloading_on_removed_file(self, mock_log):
    file_cache = {}
    path = os.path.join(self.tmpdir.path, 'tmpfile')
    reloaded, data = _ch.read_cached_file(file_cache, path)
    self.assertEqual({}, data)
    self.assertTrue(reloaded)
    mock_log.error.assert_called_once()