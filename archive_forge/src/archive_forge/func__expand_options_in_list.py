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
def _expand_options_in_list(self, slist, env=None, _ref_stack=None):
    """Expand options in  a list of strings in the configuration context.

        Args:
          slist: A list of strings.

          env: An option dict defining additional configuration options or
            overriding existing ones.

          _ref_stack: Private list containing the options being
            expanded to detect loops.

        Returns: The flatten list of expanded strings.
        """
    result = []
    for s in slist:
        value = self._expand_options_in_string(s, env, _ref_stack)
        if isinstance(value, list):
            result.extend(value)
        else:
            result.append(value)
    return result