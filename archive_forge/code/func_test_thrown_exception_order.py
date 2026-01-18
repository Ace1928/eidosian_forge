import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_thrown_exception_order(self):
    e1 = Exception('e1')
    e2 = Exception('e2')
    tasks = (('A', None), ('B', None), ('C', 'A'))
    deps = dependencies.Dependencies(tasks)
    tg = scheduler.DependencyTaskGroup(deps, DummyTask(), reverse=self.reverse_order, error_wait_time=1, aggregate_exceptions=self.aggregate_exceptions)
    task = tg()
    next(task)
    task.throw(e1)
    next(task)
    tg.error_wait_time = None
    exc = self.assertRaises(type(e2), task.throw, e2)
    self.assertIs(e2, exc)