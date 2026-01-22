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
class ExpandingUnknownOption(errors.BzrError):
    _fmt = 'Option "%(name)s" is not defined while expanding "%(string)s".'

    def __init__(self, name, string):
        self.name = name
        self.string = string