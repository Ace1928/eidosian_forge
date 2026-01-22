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
class PyBoolObjectPtr(PyLongObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyBoolObject* i.e. one of the two
    <bool> instances (Py_True/Py_False) within the process being debugged.
    """

    def proxyval(self, visited):
        if PyLongObjectPtr.proxyval(self, visited):
            return True
        else:
            return False