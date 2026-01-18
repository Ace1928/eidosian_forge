from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('code', ['def func(): pass', 'class C: pass', 'x = 1', 'x += 1', 'del x', 'import ast'])
@pytest.mark.parametrize('context', [minimal(), limited(), unsafe()])
def test_rejects_side_effect_syntax(code, context):
    with pytest.raises(SyntaxError):
        guarded_eval(code, context)