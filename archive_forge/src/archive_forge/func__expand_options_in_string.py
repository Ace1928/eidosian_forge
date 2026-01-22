import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def _expand_options_in_string(self, string, env=None, _refs=None):
    """Expand options in the string in the configuration context.

        Args:
          string: The string to be expanded.
          env: An option dict defining additional configuration options or
            overriding existing ones.
          _refs: Private list (FIFO) containing the options being expanded
            to detect loops.

        Returns: The expanded string.
        """
    if string is None:
        return None
    if _refs is None:
        _refs = []
    result = string
    expanded = True
    while expanded:
        expanded = False
        chunks = []
        for is_ref, chunk in iter_option_refs(result):
            if not is_ref:
                chunks.append(chunk)
            else:
                expanded = True
                name = chunk[1:-1]
                if name in _refs:
                    raise OptionExpansionLoop(string, _refs)
                _refs.append(name)
                value = self._expand_option(name, env, _refs)
                if value is None:
                    raise ExpandingUnknownOption(name, string)
                chunks.append(value)
                _refs.pop()
        result = ''.join(chunks)
    return result