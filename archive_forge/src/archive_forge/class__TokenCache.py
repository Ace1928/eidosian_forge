from __future__ import annotations
import re
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, Generator, Iterable, Tuple
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text.base import StyleAndTextTuples
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.styles.pygments import pygments_token_to_classname
from .base import Lexer, SimpleLexer
class _TokenCache(Dict[Tuple[str, ...], str]):
    """
    Cache that converts Pygments tokens into `prompt_toolkit` style objects.

    ``Token.A.B.C`` will be converted into:
    ``class:pygments,pygments.A,pygments.A.B,pygments.A.B.C``
    """

    def __missing__(self, key: tuple[str, ...]) -> str:
        result = 'class:' + pygments_token_to_classname(key)
        self[key] = result
        return result