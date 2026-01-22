from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import l3_conntrack_helper
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteL3ConntrackHelper(TestConntrackHelper):

    def setUp(self):
        super(TestDeleteL3ConntrackHelper, self).setUp()
        attrs = {'router_id': self.router.id}
        self.ct_helper = network_fakes.FakeL3ConntrackHelper.create_one_l3_conntrack_helper(attrs)
        self.network_client.delete_conntrack_helper = mock.Mock(return_value=None)
        self.cmd = l3_conntrack_helper.DeleteConntrackHelper(self.app, self.namespace)

    def test_delete(self):
        arglist = [self.ct_helper.router_id, self.ct_helper.id]
        verifylist = [('conntrack_helper_id', [self.ct_helper.id]), ('router', self.ct_helper.router_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_conntrack_helper.assert_called_once_with(self.ct_helper.id, self.router.id, ignore_missing=False)
        self.assertIsNone(result)

    def test_delete_error(self):
        arglist = [self.router.id, self.ct_helper.id]
        verifylist = [('conntrack_helper_id', [self.ct_helper.id]), ('router', self.router.id)]
        self.network_client.delete_conntrack_helper.side_effect = Exception('Error message')
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)