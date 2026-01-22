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
def argparse_name_to_arg(name: str) -> str:
    if name == 'warning_level':
        return '--warnlevel'
    else:
        return '--' + name.replace('_', '-')