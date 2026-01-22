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
def _autoload_commands(self) -> None:
    """Load modular command definitions."""
    all_commandset_defs = CommandSet.__subclasses__()
    existing_commandset_types = [type(command_set) for command_set in self._installed_command_sets]

    def load_commandset_by_type(commandset_types: List[Type[CommandSet]]) -> None:
        for cmdset_type in commandset_types:
            subclasses = cmdset_type.__subclasses__()
            if subclasses:
                load_commandset_by_type(subclasses)
            else:
                init_sig = inspect.signature(cmdset_type.__init__)
                if not (cmdset_type in existing_commandset_types or len(init_sig.parameters) != 1 or 'self' not in init_sig.parameters):
                    cmdset = cmdset_type()
                    self.register_command_set(cmdset)
    load_commandset_by_type(all_commandset_defs)