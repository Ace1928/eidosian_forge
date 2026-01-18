import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def visit_html_inline(self, mdnode):
    self.visit_html(mdnode)