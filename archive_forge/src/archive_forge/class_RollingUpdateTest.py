import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class RollingUpdateTest(common.HeatTestCase):

    def check_with_update(self, with_policy=False, with_diff=False):
        current = copy.deepcopy(template)
        self.current_stack = utils.parse_stack(current)
        self.current_grp = self.current_stack['group1']
        current_grp_json = self.current_grp.frozen_definition()
        prop_diff, tmpl_diff = (None, None)
        updated = tmpl_with_updt_policy() if with_policy else copy.deepcopy(template)
        if with_diff:
            res_def = updated['resources']['group1']['properties']['resource_def']
            res_def['properties']['Foo'] = 'baz'
            prop_diff = dict({'count': 2, 'resource_def': {'properties': {'Foo': 'baz'}, 'type': 'OverwrittenFnGetRefIdType'}})
        updated_stack = utils.parse_stack(updated)
        updated_grp = updated_stack['group1']
        updated_grp_json = updated_grp.t.freeze()
        tmpl_diff = updated_grp.update_template_diff(updated_grp_json, current_grp_json)
        self.current_grp._replace = mock.Mock(return_value=[])
        self.current_grp._assemble_nested = mock.Mock()
        self.patchobject(scheduler.TaskRunner, 'start')
        self.current_grp.handle_update(updated_grp_json, tmpl_diff, prop_diff)

    def test_update_without_policy_prop_diff(self):
        self.check_with_update(with_diff=True)
        self.assertTrue(self.current_grp._assemble_nested.called)

    def test_update_with_policy_prop_diff(self):
        self.check_with_update(with_policy=True, with_diff=True)
        self.current_grp._replace.assert_called_once_with(1, 2, 1)
        self.assertTrue(self.current_grp._assemble_nested.called)

    def test_update_time_not_sufficient(self):
        current = copy.deepcopy(template)
        self.stack = utils.parse_stack(current)
        self.current_grp = self.stack['group1']
        self.stack.timeout_secs = mock.Mock(return_value=200)
        err = self.assertRaises(ValueError, self.current_grp._update_timeout, 3, 100)
        self.assertIn('The current update policy will result in stack update timeout.', str(err))

    def test_update_time_sufficient(self):
        current = copy.deepcopy(template)
        self.stack = utils.parse_stack(current)
        self.current_grp = self.stack['group1']
        self.stack.timeout_secs = mock.Mock(return_value=400)
        self.assertEqual(200, self.current_grp._update_timeout(3, 100))