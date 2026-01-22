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
def _create_from_string(self, unicode_bytes, save):
    super()._create_from_string(unicode_bytes, False)
    if save:
        self.lock_write()
        self._write_config_file()
        self.unlock()