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
def alloc_string(self, string):
    pointer = self.malloc(len(string))
    get_selected_inferior().write_memory(pointer, string)
    return pointer