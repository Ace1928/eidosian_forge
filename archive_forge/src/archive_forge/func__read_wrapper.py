import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
def _read_wrapper(self, *args, **kwargs):
    function = self._readfunc()
    if function is None:
        self._warn_none('read')
        return None
    if not inspect.isgeneratorfunction(function):
        function = functools.partial(wrap_read_function, function)
    return function(*args, **kwargs)