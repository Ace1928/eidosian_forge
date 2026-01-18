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
def test_quota_list_compute_no_project_not_found(self):
    self.compute_client.quotas.get = mock.Mock(side_effect=[self.compute_quotas[0], exceptions.NotFound('NotFound')])
    arglist = ['--compute']
    verifylist = [('compute', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    ret_quotas = list(data)
    self.assertEqual(self.compute_column_header, columns)
    self.assertEqual(self.compute_reference_data, ret_quotas[0])
    self.assertEqual(1, len(ret_quotas))