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
def _set_option(self, section_name, option_name, value):
    """Set an authentication configuration option"""
    conf = self._get_config()
    section = conf.get(section_name)
    if section is None:
        conf[section_name] = {}
        section = conf[section_name]
    section[option_name] = value
    self._save()