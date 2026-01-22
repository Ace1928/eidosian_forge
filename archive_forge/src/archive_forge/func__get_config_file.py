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
def _get_config_file(self):
    try:
        f = BytesIO(self._transport.get_bytes(self._filename))
        for hook in OldConfigHooks['load']:
            hook(self)
        return f
    except transport.NoSuchFile:
        return BytesIO()
    except errors.PermissionDenied:
        trace.warning('Permission denied while trying to open configuration file %s.', urlutils.unescape_for_display(urlutils.join(self._transport.base, self._filename), 'utf-8'))
        return BytesIO()