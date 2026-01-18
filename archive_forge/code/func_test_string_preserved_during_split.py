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
def test_string_preserved_during_split(self):
    self.do_layout_test('      # The string in this command should not be split\n      set_target_properties(foo bar baz PROPERTIES COMPILE_FLAGS "-std=c++11 -Wall -Wextra")\n      ', [(NodeType.BODY, 0, 0, 0, 72, [(NodeType.COMMENT, 0, 0, 0, 48, []), (NodeType.STATEMENT, 0, 1, 0, 72, [(NodeType.FUNNAME, 0, 1, 0, 21, []), (NodeType.LPAREN, 0, 1, 21, 22, []), (NodeType.ARGGROUP, 0, 1, 22, 71, [(NodeType.PARGGROUP, 0, 1, 22, 33, [(NodeType.ARGUMENT, 0, 1, 22, 25, []), (NodeType.ARGUMENT, 0, 1, 26, 29, []), (NodeType.ARGUMENT, 0, 1, 30, 33, [])]), (NodeType.KWARGGROUP, 0, 1, 34, 71, [(NodeType.KEYWORD, 0, 1, 34, 44, []), (NodeType.ARGGROUP, 0, 1, 45, 71, [(NodeType.PARGGROUP, 0, 1, 45, 71, [(NodeType.ARGUMENT, 0, 1, 45, 58, []), (NodeType.ARGUMENT, 0, 2, 45, 71, [])])])])]), (NodeType.RPAREN, 0, 2, 71, 72, [])])])])