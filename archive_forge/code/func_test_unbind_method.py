from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_unbind_method():

    class X(list):

        def index(self, k):
            return 'CUSTOM'
    x = X()
    assert _unbind_method(x.index) is X.index
    assert _unbind_method([].index) is list.index
    assert _unbind_method(list.index) is None