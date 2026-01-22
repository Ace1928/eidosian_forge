from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import endpoint_group
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestDeleteEndpointGroup(TestEndpointGroup, common.TestDeleteVPNaaS):

    def setUp(self):
        super(TestDeleteEndpointGroup, self).setUp()
        self.networkclient.delete_vpn_endpoint_group = mock.Mock()
        self.mocked = self.networkclient.delete_vpn_endpoint_group
        self.cmd = endpoint_group.DeleteEndpointGroup(self.app, self.namespace)