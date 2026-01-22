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
class MetaParserBuilder(abc.ABCMeta):
    """Utility class wrapping base-class methods into decorators that assure read-only properties."""

    def __new__(cls, name: str, bases: Tuple, clsdict: Dict[str, Any]) -> 'MetaParserBuilder':
        """Equip all base-class methods with a needs_values decorator, and all non-const
        methods with a set_dirty_and_flush_changes decorator in addition to that.
        """
        kmm = '_mutating_methods_'
        if kmm in clsdict:
            mutating_methods = clsdict[kmm]
            for base in bases:
                methods = (t for t in inspect.getmembers(base, inspect.isroutine) if not t[0].startswith('_'))
                for name, method in methods:
                    if name in clsdict:
                        continue
                    method_with_values = needs_values(method)
                    if name in mutating_methods:
                        method_with_values = set_dirty_and_flush_changes(method_with_values)
                    clsdict[name] = method_with_values
        new_type = super().__new__(cls, name, bases, clsdict)
        return new_type