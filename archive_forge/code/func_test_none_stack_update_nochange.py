from unittest import mock
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import none_resource as none
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_none_stack_update_nochange(self):
    self._create_none_stack()
    before_refid = self.rsrc.FnGetRefId()
    self.assertIsNotNone(before_refid)
    utils.update_stack(self.stack, self.t)
    self.assertEqual((self.stack.UPDATE, self.stack.COMPLETE), self.stack.state)
    self.assertEqual(before_refid, self.stack['none'].FnGetRefId())