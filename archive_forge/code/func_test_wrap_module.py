import collections
import dataclasses
import importlib.metadata
import inspect
import logging
import multiprocessing
import os
import sys
import traceback
import types
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple
from importlib.machinery import ModuleSpec
from unittest import mock
import duet
import numpy as np
import pandas as pd
import pytest
import sympy
from _pytest.outcomes import Failed
import cirq.testing
from cirq._compat import (
def test_wrap_module():
    my_module = types.ModuleType('my_module', 'my doc string')
    my_module.foo = 'foo'
    my_module.bar = 'bar'
    my_module.__spec__ = ModuleSpec('my_module', loader=None)
    assert 'foo' in my_module.__dict__
    assert 'bar' in my_module.__dict__
    assert 'zoo' not in my_module.__dict__
    with pytest.raises(AssertionError, match='deadline should match vX.Y'):
        deprecate_attributes(my_module, {'foo': ('invalid', 'use bar instead')})
    sys.modules['my_module'] = my_module
    wrapped = deprecate_attributes('my_module', {'foo': ('v0.6', 'use bar instead')})
    assert wrapped is sys.modules.pop('my_module')
    assert wrapped.__doc__ == 'my doc string'
    assert wrapped.__name__ == 'my_module'
    assert wrapped.__spec__ is my_module.__spec__
    wrapped.__spec__ = ModuleSpec('my_module', loader=None)
    assert my_module.__spec__ is wrapped.__spec__
    assert 'foo' in wrapped.__dict__
    assert 'bar' in wrapped.__dict__
    assert 'zoo' not in wrapped.__dict__
    with cirq.testing.assert_deprecated('_compat_test.py:', 'foo was used but is deprecated.', 'will be removed in cirq v0.6', 'use bar instead', deadline='v0.6'):
        _ = wrapped.foo
    with pytest.raises(ValueError, match='During testing using Cirq deprecated functionality is not allowed'):
        _ = wrapped.foo
    with cirq.testing.assert_logs(count=0):
        _ = wrapped.bar