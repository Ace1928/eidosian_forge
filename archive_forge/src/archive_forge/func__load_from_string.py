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
def _load_from_string(self, bytes):
    """Create a config store from a string.

        Args:
          bytes: A string representing the file content.
        """
    if self.is_loaded():
        raise AssertionError('Already loaded: {!r}'.format(self._config_obj))
    co_input = BytesIO(bytes)
    try:
        self._config_obj = ConfigObj(co_input, encoding='utf-8', list_values=False)
    except configobj.ConfigObjError as e:
        self._config_obj = None
        raise ParseConfigError(e.errors, self.external_url())
    except UnicodeDecodeError:
        raise ConfigContentError(self.external_url())