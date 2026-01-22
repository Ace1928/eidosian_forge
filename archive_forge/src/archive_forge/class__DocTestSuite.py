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
class _DocTestSuite(unittest.TestSuite):

    def _removeTestAtIndex(self, index):
        pass