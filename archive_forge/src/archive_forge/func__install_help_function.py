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
def _install_help_function(self, cmd_name: str, cmd_help: Callable[..., None]) -> None:
    help_func_name = HELP_FUNC_PREFIX + cmd_name
    if hasattr(self, help_func_name):
        raise CommandSetRegistrationError(f'Attribute already exists: {help_func_name}')
    setattr(self, help_func_name, cmd_help)