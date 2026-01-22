import operator
from unittest import mock
from osc_lib.tests.utils import ParserException
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
class TestSetRouterAssoc(fakes.TestNeutronClientBgpvpn):

    def setUp(self):
        super(TestSetRouterAssoc, self).setUp()
        self.cmd = fakes.SetBgpvpnFakeRouterAssoc(self.app, self.namespace)
        self.fake_bgpvpn = fakes.create_one_bgpvpn()
        self.fake_router = fakes.create_one_resource()
        self.networkclient.find_bgpvpn = mock.Mock(side_effect=lambda name_or_id: {'id': name_or_id})

    def _build_args(self, fake_res_assoc, param=None):
        arglist_base = [fake_res_assoc['id'], self.fake_bgpvpn['id']]
        if param is not None:
            if isinstance(param, list):
                arglist_base.extend(param)
            else:
                arglist_base.append(param)
        return arglist_base

    def _build_verify_list(self, fake_res_assoc, param=None):
        verifylist = [('resource_association_id', fake_res_assoc['id']), ('bgpvpn', self.fake_bgpvpn['id'])]
        if param is not None:
            verifylist.append(param)
        return verifylist

    def test_set_router_association_no_advertise(self):
        fake_res_assoc = fakes.create_one_resource_association(self.fake_router, {'advertise_extra_routes': True})
        self.networkclient.update_bgpvpn_router_association = mock.Mock()
        arglist = self._build_args(fake_res_assoc, '--no-advertise_extra_routes')
        verifylist = [('resource_association_id', fake_res_assoc['id']), ('bgpvpn', self.fake_bgpvpn['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.networkclient.update_bgpvpn_router_association.assert_called_once_with(self.fake_bgpvpn['id'], fake_res_assoc['id'], **{'advertise_extra_routes': False})
        self.assertIsNone(result)

    def test_set_router_association_advertise(self):
        fake_res_assoc = fakes.create_one_resource_association(self.fake_router, {'advertise_extra_routes': False})
        self.networkclient.update_bgpvpn_router_association = mock.Mock()
        arglist = self._build_args(fake_res_assoc, '--advertise_extra_routes')
        verifylist = [('resource_association_id', fake_res_assoc['id']), ('bgpvpn', self.fake_bgpvpn['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.networkclient.update_bgpvpn_router_association.assert_called_once_with(self.fake_bgpvpn['id'], fake_res_assoc['id'], **{'advertise_extra_routes': True})
        self.assertIsNone(result)