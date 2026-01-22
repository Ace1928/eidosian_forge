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
def _dotall(pattern: str) -> str:
    """Makes the '.' special character match any character inside the pattern, including a newline.

    This is implemented with the inline flag `(?s:...)` and is equivalent to using `re.DOTALL`.
    It is useful for LaTeX environments, where line breaks may be present.
    """
    return f'(?s:{pattern})'