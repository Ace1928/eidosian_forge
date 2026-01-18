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
def test_validate_without_InstanceId_and_LaunchConfigurationName(self):
    t = template_format.parse(as_template)
    agp = t['Resources']['WebServerGroup']['Properties']
    agp.pop('LaunchConfigurationName')
    agp.pop('LoadBalancerNames')
    stack = utils.parse_stack(t, params=inline_templates.as_params)
    rsrc = stack['WebServerGroup']
    error_msg = "Either 'InstanceId' or 'LaunchConfigurationName' must be provided."
    exc = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    self.assertIn(error_msg, str(exc))