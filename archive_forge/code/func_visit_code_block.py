import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def visit_code_block(self, mdnode):
    kwargs = {}
    if mdnode.is_fenced and mdnode.info:
        kwargs['language'] = mdnode.info
    text = ''.join(mdnode.literal)
    if text.endswith('\n'):
        text = text[:-1]
    node = nodes.literal_block(text, text, **kwargs)
    self.current_node.append(node)