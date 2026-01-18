from unittest import mock
from zunclient.tests.unit.v1 import shell_test_base
@mock.patch('zunclient.v1.versions.VersionManager.list')
def test_zun_version_list_failure(self, mock_list):
    self._test_arg_failure('version-list --wrong', self._unrecognized_arg_error)
    self.assertFalse(mock_list.called)