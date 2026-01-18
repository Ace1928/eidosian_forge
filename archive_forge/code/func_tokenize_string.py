from __future__ import annotations
from io import StringIO
from keyword import iskeyword
import token
import tokenize
from typing import TYPE_CHECKING
def tokenize_string(source: str) -> Iterator[tuple[int, str]]:
    """
    Tokenize a Python source code string.

    Parameters
    ----------
    source : str
        The Python source code string.

    Returns
    -------
    tok_generator : Iterator[Tuple[int, str]]
        An iterator yielding all tokens with only toknum and tokval (Tuple[ing, str]).
    """
    line_reader = StringIO(source).readline
    token_generator = tokenize.generate_tokens(line_reader)
    for toknum, tokval, start, _, _ in token_generator:
        if tokval == '`':
            try:
                yield tokenize_backtick_quoted_string(token_generator, source, string_start=start[1] + 1)
            except Exception as err:
                raise SyntaxError(f"Failed to parse backticks in '{source}'.") from err
        else:
            yield (toknum, tokval)