import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
def trace_dispatch(self, *args):
    save_stdout = sys.stdout
    sys.stdout = self.__out
    try:
        return pdb.Pdb.trace_dispatch(self, *args)
    finally:
        sys.stdout = save_stdout