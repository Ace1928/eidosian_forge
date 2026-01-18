from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_set_literal():
    context = limited()
    assert guarded_eval('set()', context) == set()
    assert guarded_eval('{"a"}', context) == {'a'}