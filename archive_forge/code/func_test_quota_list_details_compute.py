import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.common import quota
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
def test_quota_list_details_compute(self):
    detailed_quota = compute_fakes.create_one_comp_detailed_quota()
    detailed_column_header = ('Resource', 'In Use', 'Reserved', 'Limit')
    detailed_reference_data = self._get_detailed_reference_data(detailed_quota)
    self.compute_client.quotas.get = mock.Mock(return_value=detailed_quota)
    arglist = ['--detail', '--compute']
    verifylist = [('detail', True), ('compute', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    ret_quotas = list(data)
    self.assertEqual(detailed_column_header, columns)
    self.assertEqual(sorted(detailed_reference_data), sorted(ret_quotas))