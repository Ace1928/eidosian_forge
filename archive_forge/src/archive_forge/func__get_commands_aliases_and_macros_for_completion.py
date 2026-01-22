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
def _get_commands_aliases_and_macros_for_completion(self) -> List[str]:
    """Return a list of visible commands, aliases, and macros for tab completion"""
    visible_commands = set(self.get_visible_commands())
    alias_names = set(self.aliases)
    macro_names = set(self.macros)
    return list(visible_commands | alias_names | macro_names)