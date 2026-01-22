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
def _expand_option(self, name, env, _refs):
    if env is not None and name in env:
        value = env[name]
    else:
        value = self.get(name, expand=False, convert=False)
        value = self._expand_options_in_string(value, env, _refs)
    return value