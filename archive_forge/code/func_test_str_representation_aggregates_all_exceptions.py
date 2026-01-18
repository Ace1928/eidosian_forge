import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_str_representation_aggregates_all_exceptions(self):
    ex1 = Exception('ex 1')
    ex2 = Exception('ex 2')
    exception_group = scheduler.ExceptionGroup([ex1, ex2])
    self.assertEqual("['ex 1', 'ex 2']", str(exception_group))