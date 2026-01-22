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
@staticmethod
def add_builtin_option(opts_map: 'MutableKeyedOptionDictType', key: OptionKey, opt: 'BuiltinOption') -> None:
    if key.subproject:
        if opt.yielding:
            return
        value = opts_map[key.as_root()].value
    else:
        value = None
    opts_map[key] = opt.init_option(key, value, default_prefix())