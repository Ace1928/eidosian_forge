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
class _SavedCmd2Env:
    """cmd2 environment settings that are backed up when entering an interactive Python shell"""

    def __init__(self) -> None:
        self.readline_settings = _SavedReadlineSettings()
        self.readline_module: Optional[ModuleType] = None
        self.history: List[str] = []
        self.sys_stdout: Optional[TextIO] = None
        self.sys_stdin: Optional[TextIO] = None