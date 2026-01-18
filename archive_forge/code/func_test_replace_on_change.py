from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as client
from heat.engine import resource
from heat.engine.resources.openstack.mistral import external_resource
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_replace_on_change(self):
    execution = self._create_resource('execution', self.rsrc_defn, self.stack)
    scheduler.TaskRunner(execution.create)()
    expected_state = (execution.CREATE, execution.COMPLETE)
    self.assertEqual(expected_state, execution.state)
    tmpl = template_format.parse(external_resource_template)
    tmpl['resources']['custom']['properties']['input']['foo2'] = '4567'
    res_defns = template.Template(tmpl).resource_definitions(self.stack)
    new_custom_defn = res_defns['custom']
    self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(execution.update, new_custom_defn))