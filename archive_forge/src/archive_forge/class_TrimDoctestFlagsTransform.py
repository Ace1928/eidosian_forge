import sys
from typing import Any, Dict, List, NamedTuple
from docutils import nodes
from docutils.nodes import Node, TextElement
from pygments.lexers import PythonConsoleLexer, guess_lexer
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext import doctest
from sphinx.transforms import SphinxTransform
class TrimDoctestFlagsTransform(SphinxTransform):
    """
    Trim doctest flags like ``# doctest: +FLAG`` from python code-blocks.

    see :confval:`trim_doctest_flags` for more information.
    """
    default_priority = HighlightLanguageTransform.default_priority + 1

    def apply(self, **kwargs: Any) -> None:
        for lbnode in self.document.findall(nodes.literal_block):
            if self.is_pyconsole(lbnode):
                self.strip_doctest_flags(lbnode)
        for dbnode in self.document.findall(nodes.doctest_block):
            self.strip_doctest_flags(dbnode)

    def strip_doctest_flags(self, node: TextElement) -> None:
        if not node.get('trim_flags', self.config.trim_doctest_flags):
            return
        source = node.rawsource
        source = doctest.blankline_re.sub('', source)
        source = doctest.doctestopt_re.sub('', source)
        node.rawsource = source
        node[:] = [nodes.Text(source)]

    @staticmethod
    def is_pyconsole(node: nodes.literal_block) -> bool:
        if node.rawsource != node.astext():
            return False
        language = node.get('language')
        if language in {'pycon', 'pycon3'}:
            return True
        elif language in {'py', 'python', 'py3', 'python3', 'default'}:
            return node.rawsource.startswith('>>>')
        elif language == 'guess':
            try:
                lexer = guess_lexer(node.rawsource)
                return isinstance(lexer, PythonConsoleLexer)
            except Exception:
                pass
        return False