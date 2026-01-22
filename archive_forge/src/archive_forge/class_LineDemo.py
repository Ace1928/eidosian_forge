import os
import re
import shlex
import sys
import pygments
from pathlib import Path
from IPython.utils.text import marquee
from IPython.utils import openpy
from IPython.utils import py3compat
class LineDemo(Demo):
    """Demo where each line is executed as a separate block.

    The input script should be valid Python code.

    This class doesn't require any markup at all, and it's meant for simple
    scripts (with no nesting or any kind of indentation) which consist of
    multiple lines of input to be executed, one at a time, as if they had been
    typed in the interactive prompt.

    Note: the input can not have *any* indentation, which means that only
    single-lines of input are accepted, not even function definitions are
    valid."""

    def reload(self):
        """Reload source from disk and initialize state."""
        self.fload()
        lines = self.fobj.readlines()
        src_b = [l for l in lines if l.strip()]
        nblocks = len(src_b)
        self.src = ''.join(lines)
        self._silent = [False] * nblocks
        self._auto = [True] * nblocks
        self.auto_all = True
        self.nblocks = nblocks
        self.src_blocks = src_b
        self.src_blocks_colored = list(map(self.highlight, self.src_blocks))
        self.reset()