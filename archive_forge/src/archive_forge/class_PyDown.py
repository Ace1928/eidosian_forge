from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class PyDown(gdb.Command):
    """Select and print the python stack frame called by this one (if any)"""

    def __init__(self):
        gdb.Command.__init__(self, 'py-down', gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

    def invoke(self, args, from_tty):
        move_in_stack(move_up=False)