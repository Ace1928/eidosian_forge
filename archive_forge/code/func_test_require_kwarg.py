import importlib
import os
import subprocess
import sys
import types
from unittest import mock
import pytest
import lazy_loader as lazy
from . import rank
from ._gaussian import gaussian
from .edges import sobel, scharr, prewitt, roberts
def test_require_kwarg():
    have_importlib_metadata = importlib.util.find_spec('importlib.metadata') is not None
    dot = '.' if have_importlib_metadata else '_'
    with mock.patch(f'importlib{dot}metadata.version') as version:
        version.return_value = '1.0.0'
        math = lazy.load('math', require='somepkg >= 2.0')
        assert isinstance(math, lazy.DelayedImportErrorModule)
        math = lazy.load('math', require='somepkg >= 1.0')
        assert math.sin(math.pi) == pytest.approx(0, 1e-06)
        math = lazy.load('math', require='somepkg >= 2.0')
        assert isinstance(math, lazy.DelayedImportErrorModule)
    with pytest.raises(ValueError):
        lazy.load('math', require='somepkg >= 1.0')