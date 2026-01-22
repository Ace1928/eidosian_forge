import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _error_location(self, msg, error=True):
    if hasattr(self, '_obj'):
        try:
            filename = inspect.getsourcefile(self._obj)
        except TypeError:
            filename = None
        msg = msg + f' in the docstring of {self._obj} in {filename}.'
    if error:
        raise ValueError(msg)
    else:
        warn(msg, stacklevel=3)