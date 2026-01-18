import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_stepping(self):
    task = DummyTask()
    task.do_step = mock.Mock(return_value=None)
    runner = scheduler.TaskRunner(task)
    runner.start()
    self.assertFalse(runner.step())
    self.assertTrue(runner)
    self.assertFalse(runner.step())
    self.assertTrue(runner.step())
    self.assertFalse(runner)
    task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
    self.assertEqual(3, task.do_step.call_count)
    self.mock_sleep.assert_not_called()