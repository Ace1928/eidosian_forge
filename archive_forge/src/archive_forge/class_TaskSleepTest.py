import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
class TaskSleepTest(common.HeatTestCase):

    def setUp(self):
        super(TaskSleepTest, self).setUp()
        scheduler.ENABLE_SLEEP = True
        self.mock_sleep = self.patchobject(eventlet, 'sleep', return_value=None)

    def test_sleep(self):
        sleep_time = 42
        runner = scheduler.TaskRunner(DummyTask())
        runner(wait_time=sleep_time)
        self.mock_sleep.assert_any_call(0)
        self.mock_sleep.assert_called_with(sleep_time)

    def test_sleep_zero(self):
        runner = scheduler.TaskRunner(DummyTask())
        runner(wait_time=0)
        self.mock_sleep.assert_called_with(0)

    def test_sleep_none(self):
        runner = scheduler.TaskRunner(DummyTask())
        runner(wait_time=None)
        self.mock_sleep.assert_not_called()