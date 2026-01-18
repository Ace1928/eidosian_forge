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
def test_restore_snapshot_other_stack(self, mock_load):
    stk1 = self._create_stack('stack_snapshot_restore_other_stack_1')
    mock_load.return_value = stk1
    snapshot1 = self.engine.stack_snapshot(self.ctx, stk1.identifier(), 'snap1')
    snapshot_id = snapshot1['id']
    mock_load.assert_called_once_with(self.ctx, stack=mock.ANY)
    mock_load.reset_mock()
    stk2 = self._create_stack('stack_snapshot_restore_other_stack_2')
    mock_load.return_value = stk2
    ex = self.assertRaises(dispatcher.ExpectedException, self.engine.stack_restore, self.ctx, stk2.identifier(), snapshot_id)
    expected = 'The Snapshot (%(snapshot)s) for Stack (%(stack)s) could not be found' % {'snapshot': snapshot_id, 'stack': stk2.name}
    self.assertEqual(exception.SnapshotNotFound, ex.exc_info[0])
    self.assertIn(expected, str(ex.exc_info[1]))
    mock_load.assert_called_once_with(self.ctx, stack=mock.ANY)