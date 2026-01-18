import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_thread_first():
    assert thread_first(2) == 2
    assert thread_first(2, inc) == 3
    assert thread_first(2, inc, inc) == 4
    assert thread_first(2, double, inc) == 5
    assert thread_first(2, (add, 5), double) == 14