import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def _initialize_history(self, hist_file: str) -> None:
    """Initialize history using history related attributes

        :param hist_file: optional path to persistent history file. If specified, then history from
                          previous sessions will be included. Additionally, all history will be written
                          to this file when the application exits.
        """
    import json
    import lzma
    self.history = History()
    if not hist_file:
        self.persistent_history_file = hist_file
        return
    hist_file = os.path.abspath(os.path.expanduser(hist_file))
    if os.path.isdir(hist_file):
        self.perror(f"Persistent history file '{hist_file}' is a directory")
        return
    hist_file_dir = os.path.dirname(hist_file)
    try:
        os.makedirs(hist_file_dir, exist_ok=True)
    except OSError as ex:
        self.perror(f"Error creating persistent history file directory '{hist_file_dir}': {ex}")
        return
    try:
        with open(hist_file, 'rb') as fobj:
            compressed_bytes = fobj.read()
        history_json = lzma.decompress(compressed_bytes).decode(encoding='utf-8')
        self.history = History.from_json(history_json)
    except FileNotFoundError:
        pass
    except OSError as ex:
        self.perror(f"Cannot read persistent history file '{hist_file}': {ex}")
        return
    except (json.JSONDecodeError, lzma.LZMAError, KeyError, UnicodeDecodeError, ValueError) as ex:
        self.perror(f"Error processing persistent history file '{hist_file}': {ex}\nThe history file will be recreated when this application exits.")
    self.history.start_session()
    self.persistent_history_file = hist_file
    if rl_type != RlType.NONE:
        last = None
        for item in self.history:
            for line in item.raw.splitlines():
                if line != last:
                    readline.add_history(line)
                    last = line
    import atexit
    atexit.register(self._persist_history)