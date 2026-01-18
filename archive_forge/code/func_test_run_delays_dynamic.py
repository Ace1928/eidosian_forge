import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_run_delays_dynamic(self):
    task = DummyTask(delays=[2, 4, 1])
    task.do_step = mock.Mock(return_value=None)
    scheduler.TaskRunner(task)()
    task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
    self.assertEqual(3, task.do_step.call_count)
    self.mock_sleep.assert_has_calls([mock.call(0), mock.call(1), mock.call(1), mock.call(1), mock.call(1), mock.call(1), mock.call(1)])
    self.assertEqual(7, self.mock_sleep.call_count)