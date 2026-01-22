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
def _pointervalue(gdbval):
    """
    Return the value of the pointer as a Python int.

    gdbval.type must be a pointer type
    """
    if gdbval.address is not None:
        return int(gdbval.address)
    else:
        return int(gdbval)