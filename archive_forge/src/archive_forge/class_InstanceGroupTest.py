import json
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class InstanceGroupTest(common.HeatTestCase):

    def setUp(self):
        super(InstanceGroupTest, self).setUp()
        self.fc = fakes_nova.FakeClient()
        self.stub_ImageConstraint_validate()
        self.stub_KeypairConstraint_validate()
        self.stub_FlavorConstraint_validate()

    def get_launch_conf_name(self, stack, ig_name):
        return stack[ig_name].properties['LaunchConfigurationName']

    def test_parse_without_update_policy(self):
        tmpl = template_format.parse(ig_tmpl_without_updt_policy)
        stack = utils.parse_stack(tmpl)
        stack.validate()
        grp = stack['JobServerGroup']
        self.assertFalse(grp.update_policy['RollingUpdate'])

    def test_parse_with_update_policy(self):
        tmpl = template_format.parse(ig_tmpl_with_updt_policy)
        stack = utils.parse_stack(tmpl)
        stack.validate()
        grp = stack['JobServerGroup']
        self.assertTrue(grp.update_policy)
        self.assertEqual(1, len(grp.update_policy))
        self.assertIn('RollingUpdate', grp.update_policy)
        policy = grp.update_policy['RollingUpdate']
        self.assertIsNotNone(policy)
        self.assertGreater(len(policy), 0)
        self.assertEqual(1, int(policy['MinInstancesInService']))
        self.assertEqual(2, int(policy['MaxBatchSize']))
        self.assertEqual('PT1S', policy['PauseTime'])

    def test_parse_with_default_update_policy(self):
        tmpl = template_format.parse(ig_tmpl_with_default_updt_policy)
        stack = utils.parse_stack(tmpl)
        stack.validate()
        grp = stack['JobServerGroup']
        self.assertTrue(grp.update_policy)
        self.assertEqual(1, len(grp.update_policy))
        self.assertIn('RollingUpdate', grp.update_policy)
        policy = grp.update_policy['RollingUpdate']
        self.assertIsNotNone(policy)
        self.assertGreater(len(policy), 0)
        self.assertEqual(0, int(policy['MinInstancesInService']))
        self.assertEqual(1, int(policy['MaxBatchSize']))
        self.assertEqual('PT0S', policy['PauseTime'])

    def test_parse_with_bad_update_policy(self):
        tmpl = template_format.parse(ig_tmpl_with_bad_updt_policy)
        stack = utils.parse_stack(tmpl)
        self.assertRaises(exception.StackValidationFailed, stack.validate)

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

    def validate_update_policy_diff(self, current, updated):
        current_tmpl = template_format.parse(current)
        current_stack = utils.parse_stack(current_tmpl)
        current_grp = current_stack['JobServerGroup']
        current_snippets = dict(((n, r.frozen_definition()) for n, r in current_stack.items()))
        current_grp_json = current_snippets[current_grp.name]
        updated_tmpl = template_format.parse(updated)
        updated_stack = utils.parse_stack(updated_tmpl)
        updated_grp = updated_stack['JobServerGroup']
        updated_grp_json = updated_grp.t.freeze()
        tmpl_diff = updated_grp.update_template_diff(updated_grp_json, current_grp_json)
        self.assertTrue(tmpl_diff.update_policy_changed())
        current_grp._try_rolling_update = mock.MagicMock()
        current_grp.resize = mock.MagicMock()
        current_grp.handle_update(updated_grp_json, tmpl_diff, None)
        self.assertEqual(updated_grp_json._update_policy or {}, current_grp.update_policy.data)

    def test_update_policy_added(self):
        self.validate_update_policy_diff(ig_tmpl_without_updt_policy, ig_tmpl_with_updt_policy)

    def test_update_policy_updated(self):
        updt_template = json.loads(ig_tmpl_with_updt_policy)
        grp = updt_template['Resources']['JobServerGroup']
        policy = grp['UpdatePolicy']['RollingUpdate']
        policy['MinInstancesInService'] = '2'
        policy['MaxBatchSize'] = '4'
        policy['PauseTime'] = 'PT1M30S'
        self.validate_update_policy_diff(ig_tmpl_with_updt_policy, json.dumps(updt_template))

    def test_update_policy_removed(self):
        self.validate_update_policy_diff(ig_tmpl_with_updt_policy, ig_tmpl_without_updt_policy)