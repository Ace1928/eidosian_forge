from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
class UserStdOption(UserComboOption):
    """
    UserOption specific to c_std and cpp_std options. User can set a list of
    STDs in preference order and it selects the first one supported by current
    compiler.

    For historical reasons, some compilers (msvc) allowed setting a GNU std and
    silently fell back to C std. This is now deprecated. Projects that support
    both GNU and MSVC compilers should set e.g. c_std=gnu11,c11.

    This is not using self.deprecated mechanism we already have for project
    options because we want to print a warning if ALL values are deprecated, not
    if SOME values are deprecated.
    """

    def __init__(self, lang: str, all_stds: T.List[str]) -> None:
        self.lang = lang.lower()
        self.all_stds = ['none'] + all_stds
        self.deprecated_stds: T.Dict[str, str] = {}
        super().__init__(f'{lang} language standard to use', ['none'], 'none')

    def set_versions(self, versions: T.List[str], gnu: bool=False, gnu_deprecated: bool=False) -> None:
        assert all((std in self.all_stds for std in versions))
        self.choices += versions
        if gnu:
            gnu_stds_map = {f'gnu{std[1:]}': std for std in versions}
            if gnu_deprecated:
                self.deprecated_stds.update(gnu_stds_map)
            else:
                self.choices += gnu_stds_map.keys()

    def validate_value(self, value: T.Union[str, T.List[str]]) -> str:
        candidates = UserArrayOption.listify_value(value)
        unknown = [std for std in candidates if std not in self.all_stds]
        if unknown:
            raise MesonException(f'Unknown {self.lang.upper()} std {unknown}. Possible values are {self.all_stds}.')
        for std in candidates:
            if std in self.choices:
                return std
        for std in candidates:
            newstd = self.deprecated_stds.get(std)
            if newstd is not None:
                mlog.deprecation(f'None of the values {candidates} are supported by the {self.lang} compiler.\n' + f'However, the deprecated {std} std currently falls back to {newstd}.\n' + 'This will be an error in the future.\n' + 'If the project supports both GNU and MSVC compilers, a value such as\n' + '"c_std=gnu11,c11" specifies that GNU is prefered but it can safely fallback to plain c11.')
                return newstd
        raise MesonException(f'None of values {candidates} are supported by the {self.lang.upper()} compiler. ' + f'Possible values are {self.choices}')