from collections import Counter, defaultdict, deque, OrderedDict, UserList
import os
import pytest
import types
import string
import sys
import unittest
import pytest
from IPython.lib import pretty
from io import StringIO
def test_super_repr(self):
    output = pretty.pretty(super(SA))
    self.assertRegex(output, '<super: \\S+.SA, None>')
    sb = SB()
    output = pretty.pretty(super(SA, sb))
    self.assertRegex(output, '<super: \\S+.SA,\\s+<\\S+.SB at 0x\\S+>>')