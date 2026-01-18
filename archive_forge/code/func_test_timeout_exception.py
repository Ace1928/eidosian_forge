import json
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_timeout_exception(self):
    self.stub_ImageConstraint_validate()
    self.stub_KeypairConstraint_validate()
    self.stub_FlavorConstraint_validate()
    t = template_format.parse(ig_tmpl_with_updt_policy)
    stack = utils.parse_stack(t)
    defn = rsrc_defn.ResourceDefinition('asg', 'OS::Heat::InstanceGroup', {'Size': 2, 'AvailabilityZones': ['zoneb'], 'LaunchConfigurationName': 'LaunchConfig', 'LoadBalancerNames': ['ElasticLoadBalancer']})
    group = instgrp.InstanceGroup('asg', defn, stack)
    group._group_data().size = mock.Mock(return_value=12)
    group.get_size = mock.Mock(return_value=12)
    self.assertRaises(ValueError, group._replace, 10, 1, 14 * 60)