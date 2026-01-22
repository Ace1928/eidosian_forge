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
class NoSuchConfigOption(errors.BzrError):
    _fmt = 'The "%(option_name)s" configuration option does not exist.'

    def __init__(self, option_name):
        errors.BzrError.__init__(self, option_name=option_name)