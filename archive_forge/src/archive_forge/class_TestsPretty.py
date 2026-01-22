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
class TestsPretty(unittest.TestCase):

    def test_super_repr(self):
        output = pretty.pretty(super(SA))
        self.assertRegex(output, '<super: \\S+.SA, None>')
        sb = SB()
        output = pretty.pretty(super(SA, sb))
        self.assertRegex(output, '<super: \\S+.SA,\\s+<\\S+.SB at 0x\\S+>>')

    def test_long_list(self):
        lis = list(range(10000))
        p = pretty.pretty(lis)
        last2 = p.rsplit('\n', 2)[-2:]
        self.assertEqual(last2, [' 999,', ' ...]'])

    def test_long_set(self):
        s = set(range(10000))
        p = pretty.pretty(s)
        last2 = p.rsplit('\n', 2)[-2:]
        self.assertEqual(last2, [' 999,', ' ...}'])

    def test_long_tuple(self):
        tup = tuple(range(10000))
        p = pretty.pretty(tup)
        last2 = p.rsplit('\n', 2)[-2:]
        self.assertEqual(last2, [' 999,', ' ...)'])

    def test_long_dict(self):
        d = {n: n for n in range(10000)}
        p = pretty.pretty(d)
        last2 = p.rsplit('\n', 2)[-2:]
        self.assertEqual(last2, [' 999: 999,', ' ...}'])

    def test_unbound_method(self):
        output = pretty.pretty(MyObj.somemethod)
        self.assertIn('MyObj.somemethod', output)