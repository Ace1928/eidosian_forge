import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
def add_subscriber(self, subscriber, precedence=0):
    """
        Register a callable subscriber to this stream which will be
        invoked either when event is called or when this stream is
        passed to the trigger classmethod.

        Precedence allows the subscriber ordering to be
        controlled. Users should only add subscribers with precedence
        between zero and one while HoloViews itself reserves the use of
        higher precedence values. Subscribers with high precedence are
        invoked later than ones with low precedence.
        """
    if not callable(subscriber):
        raise TypeError('Subscriber must be a callable.')
    self._subscribers.append((precedence, subscriber))