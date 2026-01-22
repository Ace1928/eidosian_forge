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
@property
def _current_script_dir(self) -> Optional[str]:
    """Accessor to get the current script directory from the _script_dir LIFO queue."""
    if self._script_dir:
        return self._script_dir[-1]
    else:
        return None