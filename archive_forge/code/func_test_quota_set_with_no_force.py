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
def test_quota_set_with_no_force(self):
    arglist = ['--subnets', str(network_fakes.QUOTA['subnet']), '--volumes', str(volume_fakes.QUOTA['volumes']), '--cores', str(compute_fakes.core_num), '--no-force', self.projects[0].name]
    verifylist = [('subnet', network_fakes.QUOTA['subnet']), ('volumes', volume_fakes.QUOTA['volumes']), ('cores', compute_fakes.core_num), ('force', False), ('project', self.projects[0].name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs_compute = {'cores': compute_fakes.core_num, 'force': False}
    kwargs_volume = {'volumes': volume_fakes.QUOTA['volumes']}
    kwargs_network = {'subnet': network_fakes.QUOTA['subnet'], 'check_limit': True}
    self.compute_quotas_mock.update.assert_called_once_with(self.projects[0].id, **kwargs_compute)
    self.volume_quotas_mock.update.assert_called_once_with(self.projects[0].id, **kwargs_volume)
    self.network_client.update_quota.assert_called_once_with(self.projects[0].id, **kwargs_network)
    self.assertIsNone(result)