import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
def _specified_or_unicode_argv(argv):
    if argv is None:
        return sys.argv[1:]
    new_argv = []
    try:
        for a in argv:
            if not isinstance(a, str):
                raise ValueError('not native str or unicode: {!r}'.format(a))
            new_argv.append(a)
    except (ValueError, UnicodeDecodeError):
        raise errors.BzrError('argv should be list of unicode strings.')
    return new_argv