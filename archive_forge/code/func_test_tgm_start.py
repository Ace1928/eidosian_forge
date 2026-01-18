from unittest import mock
import eventlet
from oslo_context import context
from heat.engine import service
from heat.tests import common
def test_tgm_start(self):
    stack_id = 'test'
    thm = service.ThreadGroupManager()
    ret = thm.start(stack_id, self.f, *self.fargs, **self.fkwargs)
    self.assertEqual(self.tg_mock, thm.groups['test'])
    self.tg_mock.add_thread.assert_called_with(thm._start_with_trace, context.get_current(), None, self.f, *self.fargs, **self.fkwargs)
    self.assertEqual(ret, self.tg_mock.add_thread())