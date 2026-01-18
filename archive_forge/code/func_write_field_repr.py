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
def write_field_repr(self, name, out, visited):
    """
        Extract the PyObject* field named "name", and write its representation
        to file-like object "out"
        """
    field_obj = self.pyop_field(name)
    field_obj.write_repr(out, visited)