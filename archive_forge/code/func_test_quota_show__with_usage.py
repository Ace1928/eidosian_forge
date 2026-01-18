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
def test_quota_show__with_usage(self):
    self.compute_quota = compute_fakes.create_one_comp_detailed_quota()
    self.compute_quotas_mock.get.return_value = self.compute_quota
    self.volume_quota = volume_fakes.create_one_detailed_quota()
    self.volume_quotas_mock.get.return_value = self.volume_quota
    self.network_client.get_quota.return_value = network_fakes.FakeQuota.create_one_net_detailed_quota()
    arglist = ['--usage', self.projects[0].name]
    verifylist = [('usage', True), ('project', self.projects[0].name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.compute_quotas_mock.get.assert_called_once_with(self.projects[0].id, detail=True)
    self.volume_quotas_mock.get.assert_called_once_with(self.projects[0].id, usage=True)
    self.network_client.get_quota.assert_called_once_with(self.projects[0].id, details=True)