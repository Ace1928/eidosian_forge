from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import template_format
from heat.engine import service
from heat.engine import stack
from heat.objects import snapshot as snapshot_objects
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(stack.Stack, 'load')
def test_restore_snapshot(self, mock_load):
    stk = self._create_stack('stack_snapshot_restore_normal')
    mock_load.return_value = stk
    snapshot = self.engine.stack_snapshot(self.ctx, stk.identifier(), 'snap1')
    snapshot_id = snapshot['id']
    self.engine.stack_restore(self.ctx, stk.identifier(), snapshot_id)
    self.assertEqual((stk.RESTORE, stk.COMPLETE), stk.state)
    self.assertEqual(2, mock_load.call_count)