from __future__ import annotations
from collections.abc import Generator, Sequence
import textwrap
from typing import Any, NamedTuple, TypeVar, overload
from .token import Token
def _set_children_from_tokens(self, tokens: Sequence[Token]) -> None:
    """Convert the token stream to a tree structure and set the resulting
        nodes as children of `self`."""
    reversed_tokens = list(reversed(tokens))
    while reversed_tokens:
        token = reversed_tokens.pop()
        if not token.nesting:
            self._add_child([token])
            continue
        if token.nesting != 1:
            raise ValueError('Invalid token nesting')
        nested_tokens = [token]
        nesting = 1
        while reversed_tokens and nesting:
            token = reversed_tokens.pop()
            nested_tokens.append(token)
            nesting += token.nesting
        if nesting:
            raise ValueError(f'unclosed tokens starting {nested_tokens[0]}')
        self._add_child(nested_tokens)