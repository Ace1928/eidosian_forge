import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
@needs_values
@set_dirty_and_flush_changes
def add_value(self, section: str, option: str, value: Union[str, bytes, int, float, bool]) -> 'GitConfigParser':
    """Add a value for the given option in section.

        This will create the section if required, and will not throw as opposed to the default
        ConfigParser 'set' method. The value becomes the new value of the option as returned
        by 'get_value', and appends to the list of values returned by 'get_values`'.

        :param section: Name of the section in which the option resides or should reside
        :param option: Name of the option
        :param value: Value to add to option. It must be a string or convertible
            to a string
        :return: This instance
        """
    if not self.has_section(section):
        self.add_section(section)
    self._sections[section].add(option, self._value_to_string(value))
    return self