import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_run_as_task_progress_exception_swallow(self):

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
    tr = scheduler.TaskRunner(task)
    rt = tr.as_task(progress_callback=progress)
    next(rt)
    next(rt)
    self.assertRaises(StopIteration, next, rt)
    self.assertEqual(2, len(progress_count))