from __future__ import annotations
from . import _pathlib
import sys
import os.path
import platform
import importlib
import argparse
import typing as T
from .utils.core import MesonException, MesonBugException
from . import mlog
def add_help_arguments(self, parser: argparse.ArgumentParser) -> None:
    parser.add_argument('command', nargs='?', choices=list(self.commands.keys()))