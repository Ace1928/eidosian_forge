import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_start_run_progress_exception_swallow(self):

    class TestException(Exception):
        pass
    progress_count = []

    def progress():
        try:
            if not progress_count:
                raise TestException
        finally:
            progress_count.append(None)

    def task():
        yield
        try:
            yield
        except TestException:
            yield
    runner = scheduler.TaskRunner(task)
    runner.start()
    runner.run_to_completion(progress_callback=progress)
    self.assertEqual(2, len(progress_count))
    self.mock_sleep.assert_has_calls([mock.call(1), mock.call(1)])
    self.assertEqual(2, self.mock_sleep.call_count)