import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_start_run_progress(self):
    progress_count = []

    def progress():
        progress_count.append(None)
    task = DummyTask()
    task.do_step = mock.Mock(return_value=None)
    runner = scheduler.TaskRunner(task)
    runner.start()
    runner.run_to_completion(progress_callback=progress)
    self.assertEqual(task.num_steps - 1, len(progress_count))
    task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
    self.assertEqual(3, task.do_step.call_count)
    self.mock_sleep.assert_has_calls([mock.call(1), mock.call(1)])
    self.assertEqual(2, self.mock_sleep.call_count)