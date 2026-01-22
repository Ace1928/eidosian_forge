import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def add_parsed_nodes(self, tree_nodes, keyword_token_indents):
    old_prefix = self.prefix
    tree_nodes = self._remove_endmarker(tree_nodes)
    if not tree_nodes:
        self.prefix = old_prefix + self.prefix
        return
    assert tree_nodes[0].type != 'newline'
    node = self._update_insertion_node(tree_nodes[0].start_pos[1])
    assert node.tree_node.type in ('suite', 'file_input')
    node.add_tree_nodes(old_prefix, tree_nodes)
    self._update_parsed_node_tos(tree_nodes[-1], keyword_token_indents)