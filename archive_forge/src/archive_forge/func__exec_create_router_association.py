import operator
from unittest import mock
from osc_lib.tests.utils import ParserException
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def _exec_create_router_association(self, fake_res_assoc, arglist, verifylist):
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    cols, data = self.cmd.take_action(parsed_args)
    fake_res_assoc_call = {'fake_resource_id': 'fake_resource_id', 'tenant_id': 'fake_project_id'}
    for key, value in verifylist:
        if value not in fake_res_assoc_call.values():
            fake_res_assoc_call[key] = value
    fake_res_assoc_call.pop('bgpvpn')
    self.networkclient.create_bgpvpn_router_association.assert_called_once_with(self.fake_bgpvpn['id'], **fake_res_assoc_call)
    return (cols, data)