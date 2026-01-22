import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
class TestDeleteBgpvpn(fakes.TestNeutronClientBgpvpn):

    def setUp(self):
        super(TestDeleteBgpvpn, self).setUp()
        self.networkclient.find_bgpvpn = mock.Mock(side_effect=lambda name_or_id: {'id': name_or_id})
        self.cmd = bgpvpn.DeleteBgpvpn(self.app, self.namespace)

    def test_delete_one_bgpvpn(self):
        fake_bgpvpn = fakes.create_one_bgpvpn()
        self.networkclient.delete_bgpvpn = mock.Mock()
        arglist = [fake_bgpvpn['id']]
        verifylist = [('bgpvpns', [fake_bgpvpn['id']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.networkclient.delete_bgpvpn.assert_called_once_with(fake_bgpvpn['id'])
        self.assertIsNone(result)

    def test_delete_multi_bpgvpn(self):
        fake_bgpvpns = fakes.create_bgpvpns(count=3)
        fake_bgpvpn_ids = [fake_bgpvpn['id'] for fake_bgpvpn in fake_bgpvpns]
        self.networkclient.delete_bgpvpn = mock.Mock()
        arglist = fake_bgpvpn_ids
        verifylist = [('bgpvpns', fake_bgpvpn_ids)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.networkclient.delete_bgpvpn.assert_has_calls([mock.call(id) for id in fake_bgpvpn_ids])
        self.assertIsNone(result)

    def test_delete_multi_bpgvpn_with_unknown(self):
        count = 3
        fake_bgpvpns = fakes.create_bgpvpns(count=count)
        fake_bgpvpn_ids = [fake_bgpvpn['id'] for fake_bgpvpn in fake_bgpvpns]

        def raise_unknonw_resource(resource_path, name_or_id):
            if str(count - 2) in name_or_id:
                raise Exception()
        self.networkclient.delete_bgpvpn = mock.Mock(side_effect=raise_unknonw_resource)
        arglist = fake_bgpvpn_ids
        verifylist = [('bgpvpns', fake_bgpvpn_ids)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.networkclient.delete_bgpvpn.assert_has_calls([mock.call(id) for id in fake_bgpvpn_ids])