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
def _get_nickname(self):
    value = self._get_explicit_nickname()
    if value is not None:
        return value
    if self.branch.name:
        return self.branch.name
    return urlutils.unescape(self.branch.base.split('/')[-2])