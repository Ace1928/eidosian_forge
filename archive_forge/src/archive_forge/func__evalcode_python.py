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
def _evalcode_python(executor, code, input_type):
    """
    Execute Python code in the most recent stack frame.
    """
    global_dict = gdb.parse_and_eval('PyEval_GetGlobals()')
    local_dict = gdb.parse_and_eval('PyEval_GetLocals()')
    if pointervalue(global_dict) == 0 or pointervalue(local_dict) == 0:
        raise gdb.GdbError('Unable to find the locals or globals of the most recent Python function (relative to the selected frame).')
    return executor.evalcode(code, input_type, global_dict, local_dict)