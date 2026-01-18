import json
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_parse_with_bad_pausetime_in_update_policy(self):
    tmpl = template_format.parse(ig_tmpl_with_updt_policy)
    group = tmpl['Resources']['JobServerGroup']
    policy = group['UpdatePolicy']['RollingUpdate']
    policy['PauseTime'] = 'ABCD1234'
    stack = utils.parse_stack(tmpl)
    self.assertRaises(exception.StackValidationFailed, stack.validate)
    policy['PauseTime'] = 'P1YT1H'
    stack = utils.parse_stack(tmpl)
    self.assertRaises(exception.StackValidationFailed, stack.validate)