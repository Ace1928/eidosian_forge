import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
class NeverEquals(object):
    """useful to test correct __eq__ implementation of other objects"""

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True