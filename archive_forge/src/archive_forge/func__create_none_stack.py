from unittest import mock
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import none_resource as none
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _create_none_stack(self):
    self.t = template_format.parse(self.tmpl)
    self.stack = utils.parse_stack(self.t)
    self.rsrc = self.stack['none']
    self.assertIsNone(self.rsrc.validate())
    self.stack.create()
    self.assertEqual(self.rsrc.CREATE, self.rsrc.action)
    self.assertEqual(self.rsrc.COMPLETE, self.rsrc.status)
    self.assertEqual(self.stack.CREATE, self.stack.action)
    self.assertEqual(self.stack.COMPLETE, self.stack.status)
    self.stack._update_all_resource_data(False, True)
    self.assertIsNone(self.stack.outputs['anything'].get_value())