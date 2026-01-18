import sys
from typing import Any, Dict, List, NamedTuple
from docutils import nodes
from docutils.nodes import Node, TextElement
from pygments.lexers import PythonConsoleLexer, guess_lexer
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext import doctest
from sphinx.transforms import SphinxTransform
def visit_start_of_file(self, node: Node) -> None:
    self.settings.append(self.default_setting)