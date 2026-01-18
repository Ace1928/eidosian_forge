import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_networks
def test_remove_security_service_from_share_nw_object(self):
    security_service = self._FakeSecurityService()
    share_nw = self._FakeShareNetwork()
    expected_path = (share_networks.RESOURCE_PATH + '/action') % share_nw.id
    expected_body = {'remove_security_service': {'security_service_id': security_service.id}}
    with mock.patch.object(self.manager, '_create', mock.Mock()):
        self.manager.remove_security_service(share_nw, security_service)
        self.manager._create.assert_called_once_with(expected_path, expected_body, share_networks.RESOURCE_NAME)