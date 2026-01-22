import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _copy_from_old_parser(self, line_offset, start_line_old, until_line_old, until_line_new):
    last_until_line = -1
    while until_line_new > self._nodes_tree.parsed_until_line:
        parsed_until_line_old = self._nodes_tree.parsed_until_line - line_offset
        line_stmt = self._get_old_line_stmt(parsed_until_line_old + 1)
        if line_stmt is None:
            self._parse(self._nodes_tree.parsed_until_line + 1)
        else:
            p_children = line_stmt.parent.children
            index = p_children.index(line_stmt)
            if start_line_old == 1 and p_children[0].get_first_leaf().prefix.startswith(BOM_UTF8_STRING):
                copied_nodes = []
            else:
                from_ = self._nodes_tree.parsed_until_line + 1
                copied_nodes = self._nodes_tree.copy_nodes(p_children[index:], until_line_old, line_offset)
            if copied_nodes:
                self._copy_count += 1
                to = self._nodes_tree.parsed_until_line
                LOG.debug('copy old[%s:%s] new[%s:%s]', copied_nodes[0].start_pos[0], copied_nodes[-1].end_pos[0] - 1, from_, to)
            else:
                self._parse(self._nodes_tree.parsed_until_line + 1)
        assert last_until_line != self._nodes_tree.parsed_until_line, last_until_line
        last_until_line = self._nodes_tree.parsed_until_line