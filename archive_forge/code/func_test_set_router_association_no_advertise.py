import operator
from unittest import mock
from osc_lib.tests.utils import ParserException
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_set_router_association_no_advertise(self):
    fake_res_assoc = fakes.create_one_resource_association(self.fake_router, {'advertise_extra_routes': True})
    self.networkclient.update_bgpvpn_router_association = mock.Mock()
    arglist = self._build_args(fake_res_assoc, '--no-advertise_extra_routes')
    verifylist = [('resource_association_id', fake_res_assoc['id']), ('bgpvpn', self.fake_bgpvpn['id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.networkclient.update_bgpvpn_router_association.assert_called_once_with(self.fake_bgpvpn['id'], fake_res_assoc['id'], **{'advertise_extra_routes': False})
    self.assertIsNone(result)