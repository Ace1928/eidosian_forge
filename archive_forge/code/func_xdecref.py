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
def xdecref(self, pointer):
    """Decrement the reference count of a Python object in the inferior."""
    gdb.parse_and_eval('Py_DecRef((PyObject *) %d)' % pointer)