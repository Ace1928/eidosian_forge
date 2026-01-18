import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_scaling_group_create_with_instanceid(self):
    t = template_format.parse(as_template)
    agp = t['Resources']['WebServerGroup']['Properties']
    agp['InstanceId'] = '5678'
    agp.pop('LaunchConfigurationName')
    agp.pop('LoadBalancerNames')
    stack = utils.parse_stack(t, params=inline_templates.as_params)
    rsrc = stack['WebServerGroup']
    self._stub_nova_server_get()
    _config, ins_props = rsrc._get_conf_properties()
    self.assertEqual('dd619705-468a-4f7d-8a06-b84794b3561a', ins_props['ImageId'])
    self.assertEqual('test', ins_props['KeyName'])
    self.assertEqual(['hth_test'], ins_props['SecurityGroups'])
    self.assertEqual('1', ins_props['InstanceType'])