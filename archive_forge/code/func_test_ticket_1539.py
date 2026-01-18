import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_ticket_1539(self):
    dtypes = [x for x in np.sctypeDict.values() if issubclass(x, np.number) and (not issubclass(x, np.timedelta64))]
    a = np.array([], np.bool_)
    failures = []
    for x in dtypes:
        b = a.astype(x)
        for y in dtypes:
            c = a.astype(y)
            try:
                d = np.dot(b, c)
            except TypeError:
                failures.append((x, y))
            else:
                if d != 0:
                    failures.append((x, y))
    if failures:
        raise AssertionError('Failures: %r' % failures)