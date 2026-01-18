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
def unset_alias(self, alias_name):
    """Unset an existing alias."""
    with self.lock_write():
        self.reload()
        aliases = self._get_parser().get('ALIASES')
        if not aliases or alias_name not in aliases:
            raise NoSuchAlias(alias_name)
        del aliases[alias_name]
        self._write_config_file()