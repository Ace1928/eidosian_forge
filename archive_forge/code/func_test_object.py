from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_object():
    obj = object()
    context = limited(obj=obj)
    assert guarded_eval('obj.__dir__', context) == obj.__dir__