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
class SectionConstraint(Generic[T_ConfigParser]):
    """Constrains a ConfigParser to only option commands which are constrained to
    always use the section we have been initialized with.

    It supports all ConfigParser methods that operate on an option.

    :note: If used as a context manager, will release the wrapped ConfigParser.
    """
    __slots__ = ('_config', '_section_name')
    _valid_attrs_ = ('get_value', 'set_value', 'get', 'set', 'getint', 'getfloat', 'getboolean', 'has_option', 'remove_section', 'remove_option', 'options')

    def __init__(self, config: T_ConfigParser, section: str) -> None:
        self._config = config
        self._section_name = section

    def __del__(self) -> None:
        self._config.release()

    def __getattr__(self, attr: str) -> Any:
        if attr in self._valid_attrs_:
            return lambda *args, **kwargs: self._call_config(attr, *args, **kwargs)
        return super().__getattribute__(attr)

    def _call_config(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call the configuration at the given method which must take a section name
        as first argument."""
        return getattr(self._config, method)(self._section_name, *args, **kwargs)

    @property
    def config(self) -> T_ConfigParser:
        """return: ConfigParser instance we constrain"""
        return self._config

    def release(self) -> None:
        """Equivalent to GitConfigParser.release(), which is called on our underlying
        parser instance."""
        return self._config.release()

    def __enter__(self) -> 'SectionConstraint[T_ConfigParser]':
        self._config.__enter__()
        return self

    def __exit__(self, exception_type: str, exception_value: str, traceback: str) -> None:
        self._config.__exit__(exception_type, exception_value, traceback)