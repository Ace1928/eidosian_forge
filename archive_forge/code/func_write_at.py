from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
def write_at(self, cursor, text):
    if sys.version_info[0] < 3 and isinstance(text, str):
        text = text.decode('utf-8')
    self.assert_lt(cursor)
    rows = cursor[0] - self._cursor[0]
    if rows:
        self._fobj.write(self._config.format.endl * rows)
        self._cursor[0] += rows
        self._cursor[1] = 0
    cols = cursor[1] - self._cursor[1]
    if cols:
        self._fobj.write(' ' * cols)
        self._cursor[1] += cols
    lines = text.split('\n')
    line = lines.pop(0)
    self._fobj.write(line)
    self._cursor[1] += len(line)
    while lines:
        self._fobj.write(self._config.format.endl)
        self._cursor[0] += 1
        self._cursor[1] = 0
        line = lines.pop(0)
        self._fobj.write(line)
        self._cursor[1] += len(line)