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
def _persist_history(self) -> None:
    """Write history out to the persistent history file as compressed JSON"""
    import lzma
    if not self.persistent_history_file:
        return
    self.history.truncate(self._persistent_history_length)
    try:
        history_json = self.history.to_json()
        compressed_bytes = lzma.compress(history_json.encode(encoding='utf-8'))
        with open(self.persistent_history_file, 'wb') as fobj:
            fobj.write(compressed_bytes)
    except OSError as ex:
        self.perror(f"Cannot write persistent history file '{self.persistent_history_file}': {ex}")