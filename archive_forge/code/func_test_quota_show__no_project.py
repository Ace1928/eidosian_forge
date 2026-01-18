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
def test_quota_show__no_project(self):
    arglist = []
    verifylist = [('project', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.compute_quotas_mock.get.assert_called_once_with(identity_fakes.project_id, detail=False)
    self.volume_quotas_mock.get.assert_called_once_with(identity_fakes.project_id, usage=False)
    self.network_client.get_quota.assert_called_once_with(identity_fakes.project_id, details=False)
    self.assertNotCalled(self.network_client.get_quota_default)