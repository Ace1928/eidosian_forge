from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def _set_mock_microversion(self, mock_v):
    """Set a specific microversion for the mock supports_microversion()."""
    self.supports_microversion_mock.reset_mock(return_value=True)
    self.supports_microversion_mock.side_effect = lambda _, v: api_versions.APIVersion(v) <= api_versions.APIVersion(mock_v)