from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
@mock.patch('zunclient.api_versions.APIVersion')
def test_only_major_part_is_presented(self, mock_apiversion):
    version = 7
    self.assertEqual(mock_apiversion.return_value, api_versions.get_api_version(version))
    mock_apiversion.assert_called_once_with('%s.latest' % str(version))