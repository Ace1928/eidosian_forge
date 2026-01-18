import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def visit_code(self, mdnode):
    n = nodes.literal(mdnode.literal, mdnode.literal)
    self.current_node.append(n)