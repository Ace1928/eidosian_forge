import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_run_as_task_exception(self):

    class TestException(Exception):
        pass
    task = DummyTask()
    task.do_step = mock.Mock(return_value=None)
    tr = scheduler.TaskRunner(task)
    rt = tr.as_task()
    next(rt)
    self.assertRaises(TestException, rt.throw, TestException)
    self.assertTrue(tr.done())
    task.do_step.assert_called_once_with(1)
    self.mock_sleep.assert_not_called()