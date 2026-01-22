import builtins as builtin_mod
import sys
import types
from pathlib import Path
from . import tools
from IPython.core import page
from IPython.utils import io
from IPython.terminal.interactiveshell import TerminalInteractiveShell
def _showtraceback(self, etype, evalue, stb):
    """Print the traceback purely on stdout for doctest to capture it.
    """
    print(self.InteractiveTB.stb2text(stb), file=sys.stdout)