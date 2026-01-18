import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def visit_item(self, mdnode):
    node = nodes.list_item()
    node.line = mdnode.sourcepos[0][0]
    self.current_node.append(node)
    self.current_node = node