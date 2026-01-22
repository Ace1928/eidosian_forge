from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
class BadNamedTuple(NamedTuple):
    a: str

    def __getitem__(self, key):
        return None