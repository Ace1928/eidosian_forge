import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_run_progress_exception_swallow(self):

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
        try:
            yield
        except TestException:
            yield
    scheduler.TaskRunner(task)(progress_callback=progress)
    self.assertEqual(2, len(progress_count))
    self.mock_sleep.assert_has_calls([mock.call(0), mock.call(1)])
    self.assertEqual(2, self.mock_sleep.call_count)