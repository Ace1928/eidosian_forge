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
def update_project_options(self, options: 'MutableKeyedOptionDictType') -> None:
    for key, value in options.items():
        if not key.is_project():
            continue
        if key not in self.options:
            self.options[key] = value
            continue
        oldval = self.options[key]
        if type(oldval) is not type(value):
            self.options[key] = value
        elif oldval.choices != value.choices:
            self.options[key] = value
            try:
                value.set_value(oldval.value)
            except MesonException:
                mlog.warning(f'Old value(s) of {key} are no longer valid, resetting to default ({value.value}).', fatal=False)