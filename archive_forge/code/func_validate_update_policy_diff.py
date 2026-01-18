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
def validate_update_policy_diff(self, current, updated):
    current_stack = utils.parse_stack(current)
    current_grp = current_stack['group1']
    current_grp_json = current_grp.frozen_definition()
    updated_stack = utils.parse_stack(updated)
    updated_grp = updated_stack['group1']
    updated_grp_json = updated_grp.t.freeze()
    tmpl_diff = updated_grp.update_template_diff(updated_grp_json, current_grp_json)
    self.assertTrue(tmpl_diff.update_policy_changed())
    prop_diff = current_grp.update_template_diff_properties(updated_grp.properties, current_grp.properties)
    current_grp._try_rolling_update = mock.Mock()
    current_grp._assemble_nested_for_size = mock.Mock()
    self.patchobject(scheduler.TaskRunner, 'start')
    current_grp.handle_update(updated_grp_json, tmpl_diff, prop_diff)
    self.assertEqual(updated_grp_json._update_policy or {}, current_grp.update_policy.data)