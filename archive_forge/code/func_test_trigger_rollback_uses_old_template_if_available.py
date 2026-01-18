from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_trigger_rollback_uses_old_template_if_available(self):
    t = template_format.parse(tools.wp_template)
    prev_tmpl = templatem.Template(t)
    prev_tmpl.store(context=self.ctx)
    self.stack.prev_raw_template_id = prev_tmpl.id
    self.stack.action = self.stack.UPDATE
    self.stack.status = self.stack.FAILED
    self.stack.store()
    self.stack.converge_stack = mock.Mock()
    self.stack.rollback()
    self.assertTrue(self.stack.converge_stack.called)
    self.assertIsNone(self.stack.prev_raw_template_id)
    call_args, call_kwargs = self.stack.converge_stack.call_args
    template_used_for_rollback = call_args[0]
    self.assertEqual(prev_tmpl.id, template_used_for_rollback.id)