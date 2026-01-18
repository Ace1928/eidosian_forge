from __future__ import annotations
from io import StringIO
from keyword import iskeyword
import token
import tokenize
from typing import TYPE_CHECKING
def tokenize_backtick_quoted_string(token_generator: Iterator[tokenize.TokenInfo], source: str, string_start: int) -> tuple[int, str]:
    """
    Creates a token from a backtick quoted string.

    Moves the token_generator forwards till right after the next backtick.

    Parameters
    ----------
    token_generator : Iterator[tokenize.TokenInfo]
        The generator that yields the tokens of the source string (Tuple[int, str]).
        The generator is at the first token after the backtick (`)

    source : str
        The Python source code string.

    string_start : int
        This is the start of backtick quoted string inside the source string.

    Returns
    -------
    tok: Tuple[int, str]
        The token that represents the backtick quoted string.
        The integer is equal to BACKTICK_QUOTED_STRING (100).
    """
    for _, tokval, start, _, _ in token_generator:
        if tokval == '`':
            string_end = start[1]
            break
    return (BACKTICK_QUOTED_STRING, source[string_start:string_end])