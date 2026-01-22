from __future__ import annotations
import collections
import configparser
import copy
import os
import os.path
import re
from typing import (
from coverage.exceptions import ConfigError
from coverage.misc import isolate_module, human_sorted_items, substitute_variables
from coverage.tomlconfig import TomlConfigParser, TomlDecodeError
from coverage.types import (
def _set_attr_from_config_option(self, cp: TConfigParser, attr: str, where: str, type_: str='') -> bool:
    """Set an attribute on self if it exists in the ConfigParser.

        Returns True if the attribute was set.

        """
    section, option = where.split(':')
    if cp.has_option(section, option):
        method = getattr(cp, 'get' + type_)
        setattr(self, attr, method(section, option))
        return True
    return False