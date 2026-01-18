import builtins as builtin_mod
import sys
import types
from pathlib import Path
from . import tools
from IPython.core import page
from IPython.utils import io
from IPython.terminal.interactiveshell import TerminalInteractiveShell
Start a global IPython shell, which we need for IPython-specific syntax.
    