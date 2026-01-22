from unittest import mock
import uuid
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import vpnservice
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestDeleteVPNService(TestVPNService, common.TestDeleteVPNaaS):

    def setUp(self):
        super(TestDeleteVPNService, self).setUp()
        self.networkclient.delete_vpn_service = mock.Mock()
        self.mocked = self.networkclient.delete_vpn_service
        self.cmd = vpnservice.DeleteVPNService(self.app, self.namespace)