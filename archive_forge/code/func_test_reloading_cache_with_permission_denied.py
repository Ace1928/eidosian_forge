import os
from unittest import mock
import fixtures
import oslo_config
from oslotest import base as test_base
from oslo_policy import _cache_handler as _ch
@mock.patch.object(_ch, 'LOG')
def test_reloading_cache_with_permission_denied(self, mock_log):
    file_cache = {}
    path = os.path.join(self.tmpdir.path, 'tmpfile')
    with open(path, 'w+') as fp:
        fp.write('test')
    os.chmod(path, 0)
    self.assertRaises(oslo_config.cfg.ConfigFilesPermissionDeniedError, _ch.read_cached_file, file_cache, path)
    mock_log.error.assert_called_once()