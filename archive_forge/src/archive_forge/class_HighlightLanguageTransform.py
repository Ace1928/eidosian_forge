import sys
from typing import Any, Dict, List, NamedTuple
from docutils import nodes
from docutils.nodes import Node, TextElement
from pygments.lexers import PythonConsoleLexer, guess_lexer
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext import doctest
from sphinx.transforms import SphinxTransform
class HighlightLanguageTransform(SphinxTransform):
    """
    Apply highlight_language to all literal_block nodes.

    This refers both :confval:`highlight_language` setting and
    :rst:dir:`highlightlang` directive.  After processing, this transform
    removes ``highlightlang`` node from doctree.
    """
    default_priority = 400

    def apply(self, **kwargs: Any) -> None:
        visitor = HighlightLanguageVisitor(self.document, self.config.highlight_language)
        self.document.walkabout(visitor)
        for node in list(self.document.findall(addnodes.highlightlang)):
            node.parent.remove(node)