from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_rejects_forbidden():
    context = forbidden()
    with pytest.raises(GuardRejection):
        guarded_eval('1', context)