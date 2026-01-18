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
def test_quota_set_network(self):
    arglist = ['--subnets', str(network_fakes.QUOTA['subnet']), '--networks', str(network_fakes.QUOTA['network']), '--floating-ips', str(network_fakes.QUOTA['floatingip']), '--subnetpools', str(network_fakes.QUOTA['subnetpool']), '--secgroup-rules', str(network_fakes.QUOTA['security_group_rule']), '--secgroups', str(network_fakes.QUOTA['security_group']), '--routers', str(network_fakes.QUOTA['router']), '--rbac-policies', str(network_fakes.QUOTA['rbac_policy']), '--ports', str(network_fakes.QUOTA['port']), self.projects[0].name]
    verifylist = [('subnet', network_fakes.QUOTA['subnet']), ('network', network_fakes.QUOTA['network']), ('floatingip', network_fakes.QUOTA['floatingip']), ('subnetpool', network_fakes.QUOTA['subnetpool']), ('security_group_rule', network_fakes.QUOTA['security_group_rule']), ('security_group', network_fakes.QUOTA['security_group']), ('router', network_fakes.QUOTA['router']), ('rbac_policy', network_fakes.QUOTA['rbac_policy']), ('port', network_fakes.QUOTA['port']), ('project', self.projects[0].name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'subnet': network_fakes.QUOTA['subnet'], 'network': network_fakes.QUOTA['network'], 'floatingip': network_fakes.QUOTA['floatingip'], 'subnetpool': network_fakes.QUOTA['subnetpool'], 'security_group_rule': network_fakes.QUOTA['security_group_rule'], 'security_group': network_fakes.QUOTA['security_group'], 'router': network_fakes.QUOTA['router'], 'rbac_policy': network_fakes.QUOTA['rbac_policy'], 'port': network_fakes.QUOTA['port']}
    self.network_client.update_quota.assert_called_once_with(self.projects[0].id, **kwargs)
    self.assertIsNone(result)