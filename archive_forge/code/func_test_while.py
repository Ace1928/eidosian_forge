from __future__ import unicode_literals
import contextlib
import logging
import unittest
import sys
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.format import formatter
from cmakelang.parse.common import NodeType
def test_while(self):
    self.do_layout_test('      while(forbarbaz arg1 arg2, arg3)\n        message(hello ${foobarbaz})\n      endwhile()\n      ', [(NodeType.BODY, 0, 0, 0, 32, [(NodeType.FLOW_CONTROL, 0, 0, 0, 32, [(NodeType.STATEMENT, 0, 0, 0, 32, [(NodeType.FUNNAME, 0, 0, 0, 5, []), (NodeType.LPAREN, 0, 0, 5, 6, []), (NodeType.ARGGROUP, 0, 0, 6, 31, [(NodeType.PARGGROUP, 0, 0, 6, 31, [(NodeType.ARGUMENT, 0, 0, 6, 15, []), (NodeType.ARGUMENT, 0, 0, 16, 20, []), (NodeType.ARGUMENT, 0, 0, 21, 26, []), (NodeType.ARGUMENT, 0, 0, 27, 31, [])])]), (NodeType.RPAREN, 0, 0, 31, 32, [])]), (NodeType.BODY, 0, 1, 2, 29, [(NodeType.STATEMENT, 0, 1, 2, 29, [(NodeType.FUNNAME, 0, 1, 2, 9, []), (NodeType.LPAREN, 0, 1, 9, 10, []), (NodeType.ARGGROUP, 0, 1, 10, 28, [(NodeType.PARGGROUP, 0, 1, 10, 28, [(NodeType.ARGUMENT, 0, 1, 10, 15, []), (NodeType.ARGUMENT, 0, 1, 16, 28, [])])]), (NodeType.RPAREN, 0, 1, 28, 29, [])])]), (NodeType.STATEMENT, 0, 2, 0, 10, [(NodeType.FUNNAME, 0, 2, 0, 8, []), (NodeType.LPAREN, 0, 2, 8, 9, []), (NodeType.ARGGROUP, 0, 2, 9, 9, []), (NodeType.RPAREN, 0, 2, 9, 10, [])])])])])