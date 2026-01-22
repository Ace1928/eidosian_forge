import re
import sys
from contextlib import suppress
from typing import Iterable, NamedTuple, Optional
from .color import Color
from .style import Style
from .text import Text
def _ansi_tokenize(ansi_text: str) -> Iterable[_AnsiToken]:
    """Tokenize a string in to plain text and ANSI codes.

    Args:
        ansi_text (str): A String containing ANSI codes.

    Yields:
        AnsiToken: A named tuple of (plain, sgr, osc)
    """
    position = 0
    sgr: Optional[str]
    osc: Optional[str]
    for match in re_ansi.finditer(ansi_text):
        start, end = match.span(0)
        osc, sgr = match.groups()
        if start > position:
            yield _AnsiToken(ansi_text[position:start])
        if sgr:
            if sgr == '(':
                position = end + 1
                continue
            if sgr.endswith('m'):
                yield _AnsiToken('', sgr[1:-1], osc)
        else:
            yield _AnsiToken('', sgr, osc)
        position = end
    if position < len(ansi_text):
        yield _AnsiToken(ansi_text[position:])