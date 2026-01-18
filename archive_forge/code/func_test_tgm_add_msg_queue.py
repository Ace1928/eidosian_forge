from unittest import mock
import eventlet
from oslo_context import context
from heat.engine import service
from heat.tests import common
def test_tgm_add_msg_queue(self):
    stack_id = 'add_msg_queues_test'
    e1, e2 = (mock.Mock(), mock.Mock())
    thm = service.ThreadGroupManager()
    thm.add_msg_queue(stack_id, e1)
    thm.add_msg_queue(stack_id, e2)
    self.assertEqual([e1, e2], thm.msg_queues[stack_id])