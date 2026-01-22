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
def _min_indent(self, s):
    """Return the minimum indentation of any non-blank line in `s`"""
    indents = [len(indent) for indent in self._INDENT_RE.findall(s)]
    if len(indents) > 0:
        return min(indents)
    else:
        return 0