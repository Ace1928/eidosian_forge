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
class TreeConfig(IniBasedConfig):
    """Branch configuration data associated with its contents, not location"""

    def __init__(self, branch):
        self._config = branch._get_config()
        self.branch = branch

    def _get_parser(self, file=None):
        if file is not None:
            return IniBasedConfig._get_parser(file)
        return self._config._get_configobj()

    def get_option(self, name, section=None, default=None):
        with self.branch.lock_read():
            return self._config.get_option(name, section, default)

    def set_option(self, value, name, section=None):
        """Set a per-branch configuration option"""
        with self.branch.lock_write():
            self._config.set_option(value, name, section)

    def remove_option(self, option_name, section_name=None):
        with self.branch.lock_write():
            self._config.remove_option(option_name, section_name)