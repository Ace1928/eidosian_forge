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
def _PyObject_VAR_SIZE(typeobj, nitems):
    if _PyObject_VAR_SIZE._type_size_t is None:
        _PyObject_VAR_SIZE._type_size_t = gdb.lookup_type('size_t')
    return (typeobj.field('tp_basicsize') + nitems * typeobj.field('tp_itemsize') + (_sizeof_void_p() - 1) & ~(_sizeof_void_p() - 1)).cast(_PyObject_VAR_SIZE._type_size_t)