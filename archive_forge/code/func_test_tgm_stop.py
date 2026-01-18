from unittest import mock
import eventlet
from oslo_context import context
from heat.engine import service
from heat.tests import common
def test_tgm_stop(self):
    stack_id = 'test'
    done = []

    def function():
        while True:
            eventlet.sleep()

    def linked(gt, thread):
        for i in range(10):
            eventlet.sleep()
        done.append(thread)
    thm = service.ThreadGroupManager()
    thm.add_msg_queue(stack_id, mock.Mock())
    thread = thm.start(stack_id, function)
    thread.link(linked, thread)
    thm.stop(stack_id)
    self.assertIn(thread, done)
    self.assertNotIn(stack_id, thm.groups)
    self.assertNotIn(stack_id, thm.msg_queues)