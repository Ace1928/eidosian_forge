import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def assertInText(self, pattern, text):
    """
        Assert *pattern* is in *text*, ignoring any whitespace differences
        (including newlines).
        """

    def escape(c):
        if not c.isalnum() and (not c.isspace()):
            return '\\' + c
        return c
    pattern = ''.join(map(escape, pattern))
    regex = re.sub('\\s+', '\\\\s*', pattern)
    self.assertRegex(text, regex)