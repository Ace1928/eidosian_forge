from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
class SecondLevelHolder(HoldableObject, metaclass=abc.ABCMeta):
    """ A second level object holder. The primary purpose
        of such objects is to hold multiple objects with one
        default option. """

    @abc.abstractmethod
    def get_default_object(self) -> HoldableObject:
        ...