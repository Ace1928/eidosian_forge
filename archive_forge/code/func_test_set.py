from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_set():
    context = limited(data={'a', 'b'})
    assert guarded_eval('data.difference', context)