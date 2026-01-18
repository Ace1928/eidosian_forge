from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshot_instances
@ddt.data('get', 'list', 'reset_state')
def test_upsupported_microversion(self, method_name):
    unsupported_microversions = ('1.0', '2.18')
    arguments = {'instance': 'FAKE_INSTANCE'}
    if method_name in 'list':
        arguments.clear()
    for microversion in unsupported_microversions:
        microversion = api_versions.APIVersion(microversion)
        mock_microversion = mock.Mock(api_version=microversion)
        manager = share_snapshot_instances.ShareSnapshotInstanceManager(api=mock_microversion)
        method = getattr(manager, method_name)
        self.assertRaises(exceptions.UnsupportedVersion, method, **arguments)