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
class _SpoofOut(StringIO):

    def getvalue(self):
        result = StringIO.getvalue(self)
        if result and (not result.endswith('\n')):
            result += '\n'
        return result

    def truncate(self, size=None):
        self.seek(size)
        StringIO.truncate(self)