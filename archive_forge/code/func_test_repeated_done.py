import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_repeated_done(self):
    task = DummyTask(0)
    task.do_step = mock.Mock(return_value=None)
    runner = scheduler.TaskRunner(task)
    runner.start()
    self.assertTrue(runner.step())
    self.assertTrue(runner.step())
    task.do_step.assert_not_called()
    self.mock_sleep.assert_not_called()