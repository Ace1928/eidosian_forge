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
class ConfigOptionValueError(errors.BzrError):
    _fmt = 'Bad value "%(value)s" for option "%(name)s".\nSee ``brz help %(name)s``'

    def __init__(self, name, value):
        errors.BzrError.__init__(self, name=name, value=value)