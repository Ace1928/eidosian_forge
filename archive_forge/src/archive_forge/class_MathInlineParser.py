import base64
import mimetypes
import os
from html import escape
from typing import Any, Callable, Dict, Iterable, Match, Optional, Tuple
import bs4
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import Lexer
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from nbconvert.filters.strings import add_anchor
class MathInlineParser(InlineParser):
    """This interprets the content of LaTeX style math objects.

        In particular this grabs ``$$...$$``, ``\\\\[...\\\\]``, ``\\\\(...\\\\)``, ``$...$``,
        and ``\\begin{foo}...\\end{foo}`` styles for declaring mathematics. It strips
        delimiters from all these varieties, and extracts the type of environment
        in the last case (``foo`` in this example).
        """
    BLOCK_MATH_TEX = _dotall('(?<!\\\\)\\$\\$(.*?)(?<!\\\\)\\$\\$')
    BLOCK_MATH_LATEX = _dotall('(?<!\\\\)\\\\\\\\\\[(.*?)(?<!\\\\)\\\\\\\\\\]')
    INLINE_MATH_TEX = _dotall('(?<![$\\\\])\\$(.+?)(?<![$\\\\])\\$')
    INLINE_MATH_LATEX = _dotall('(?<!\\\\)\\\\\\\\\\((.*?)(?<!\\\\)\\\\\\\\\\)')
    LATEX_ENVIRONMENT = _dotall('\\\\begin\\{([a-z]*\\*?)\\}(.*?)\\\\end\\{\\1\\}')
    RULE_NAMES = ('block_math_tex', 'block_math_latex', 'inline_math_tex', 'inline_math_latex', 'latex_environment', *InlineParser.RULE_NAMES)

    def parse_block_math_tex(self, m: Match[str], state: Any) -> Tuple[str, str]:
        """Parse block text math."""
        text = m.group(0)[2:-2]
        return ('block_math', text)

    def parse_block_math_latex(self, m: Match[str], state: Any) -> Tuple[str, str]:
        """Parse block latex math ."""
        text = m.group(1)
        return ('block_math', text)

    def parse_inline_math_tex(self, m: Match[str], state: Any) -> Tuple[str, str]:
        """Parse inline tex math."""
        text = m.group(1)
        return ('inline_math', text)

    def parse_inline_math_latex(self, m: Match[str], state: Any) -> Tuple[str, str]:
        """Parse inline latex math."""
        text = m.group(1)
        return ('inline_math', text)

    def parse_latex_environment(self, m: Match[str], state: Any) -> Tuple[str, str, str]:
        """Parse a latex environment."""
        name, text = (m.group(1), m.group(2))
        return ('latex_environment', name, text)