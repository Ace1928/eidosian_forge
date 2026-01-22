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
def _show_value(self, name, directory, scope):
    conf = self._get_stack(directory, scope)
    value = conf.get(name, expand=True, convert=False)
    if value is not None:
        value = self._quote_multiline(value)
        self.outf.write('{}\n'.format(value))
    else:
        raise NoSuchConfigOption(name)