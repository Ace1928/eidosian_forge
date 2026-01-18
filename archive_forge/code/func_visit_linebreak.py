import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def visit_linebreak(self, _):
    self.current_node.append(nodes.raw('', '<br />', format='html'))