import sys
from typing import Any, Dict, List, NamedTuple
from docutils import nodes
from docutils.nodes import Node, TextElement
from pygments.lexers import PythonConsoleLexer, guess_lexer
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext import doctest
from sphinx.transforms import SphinxTransform
def visit_highlightlang(self, node: addnodes.highlightlang) -> None:
    self.settings[-1] = HighlightSetting(node['lang'], node['force'], node['linenothreshold'])