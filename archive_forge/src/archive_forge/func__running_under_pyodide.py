import sys
import re
import functools
import os
import contextlib
import warnings
import inspect
import pathlib
from typing import Any, Callable
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.exceptions import ignore_warnings # noqa:F401
def _running_under_pyodide():
    """Test if running under pyodide."""
    try:
        import pyodide_js
    except ImportError:
        return False
    else:
        return True