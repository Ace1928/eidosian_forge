import sys
from contextlib import contextmanager
from typing import (
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('data,bad', [[[1, 2, 3], 'data.append(4)'], [{'a': 1}, 'data.update()']])
def test_rejects_calls_with_side_effects(data, bad):
    context = limited(data=data)
    with pytest.raises(GuardRejection):
        guarded_eval(bad, context)