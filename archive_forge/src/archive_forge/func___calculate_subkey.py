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
def __calculate_subkey(self, type_: DependencyCacheType) -> T.Tuple[str, ...]:
    data: T.Dict[DependencyCacheType, T.List[str]] = {DependencyCacheType.PKG_CONFIG: stringlistify(self.__builtins[self.__pkg_conf_key].value), DependencyCacheType.CMAKE: stringlistify(self.__builtins[self.__cmake_key].value), DependencyCacheType.OTHER: []}
    assert type_ in data, 'Someone forgot to update subkey calculations for a new type'
    return tuple(data[type_])