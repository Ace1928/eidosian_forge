from unittest import mock
from magnumclient.osc.v1 import quotas as osc_quotas
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_quotas_list_with_no_options(self):
    arglist = []
    verifylist = [('limit', None), ('sort_key', None), ('sort_dir', None), ('marker', None), ('all_tenants', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.quotas_mock.list.assert_called_with(limit=None, sort_dir=None, sort_key=None, marker=None, all_tenants=False)