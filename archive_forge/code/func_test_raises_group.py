from __future__ import annotations
import re
import sys
from types import TracebackType
from typing import Any
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
def test_raises_group() -> None:
    with pytest.raises(ValueError, match=wrap_escape(f'Invalid argument "{TypeError()!r}" must be exception type, Matcher, or RaisesGroup.')):
        RaisesGroup(TypeError())
    with RaisesGroup(ValueError):
        raise ExceptionGroup('foo', (ValueError(),))
    with RaisesGroup(SyntaxError):
        with RaisesGroup(ValueError):
            raise ExceptionGroup('foo', (SyntaxError(),))
    with RaisesGroup(ValueError, SyntaxError):
        raise ExceptionGroup('foo', (ValueError(), SyntaxError()))
    with RaisesGroup(SyntaxError, ValueError):
        raise ExceptionGroup('foo', (ValueError(), SyntaxError()))
    with RaisesGroup(RaisesGroup(ValueError)):
        raise ExceptionGroup('foo', (ExceptionGroup('bar', (ValueError(),)),))
    with RaisesGroup(SyntaxError, RaisesGroup(ValueError), RaisesGroup(RuntimeError)):
        raise ExceptionGroup('foo', (SyntaxError(), ExceptionGroup('bar', (ValueError(),)), ExceptionGroup('', (RuntimeError(),))))
    with pytest.raises(ExceptionGroup):
        with RaisesGroup(ValueError):
            raise ExceptionGroup('', (ValueError(), ValueError()))
    with pytest.raises(ExceptionGroup):
        with RaisesGroup(ValueError):
            raise ExceptionGroup('', (RuntimeError(), ValueError()))
    with pytest.raises(ExceptionGroup):
        with RaisesGroup(ValueError, ValueError):
            raise ExceptionGroup('', (ValueError(),))
    with pytest.raises(ExceptionGroup):
        with RaisesGroup(ValueError, SyntaxError):
            raise ExceptionGroup('', (ValueError(),))
    with RaisesGroup(ValueError, strict=False):
        raise ExceptionGroup('', (ExceptionGroup('', (ValueError(),)),))
    with RaisesGroup(RaisesGroup(ValueError, strict=False)):
        raise ExceptionGroup('', (ExceptionGroup('', (ValueError(),)),))
    with RaisesGroup(RaisesGroup(ValueError, strict=False)):
        raise ExceptionGroup('', (ExceptionGroup('', (ExceptionGroup('', (ValueError(),)),)),))
    with pytest.raises(ExceptionGroup):
        with RaisesGroup(RaisesGroup(ValueError, strict=False)):
            raise ExceptionGroup('', (ValueError(),))
    with pytest.raises(ValueError, match='^You cannot specify a nested structure inside a RaisesGroup with strict=False$'):
        RaisesGroup(RaisesGroup(ValueError), strict=False)
    with pytest.raises(ValueError, match='^value error text$'):
        with RaisesGroup(ValueError, strict=False):
            raise ValueError('value error text')