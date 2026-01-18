import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_timeout_return(self):
    st = timeutils.wallclock()

    def task():
        while True:
            try:
                yield
            except scheduler.Timeout:
                return
    self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 1.5])
    runner = scheduler.TaskRunner(task)
    runner.start(timeout=1)
    self.assertTrue(runner)
    self.assertTrue(runner.step())
    self.assertFalse(runner)
    self.assertEqual(3, timeutils.wallclock.call_count)