import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def visit_emph(self, _):
    n = nodes.emphasis()
    self.current_node.append(n)
    self.current_node = n