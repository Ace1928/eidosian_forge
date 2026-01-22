from __future__ import annotations
from collections.abc import Generator, Sequence
import textwrap
from typing import Any, NamedTuple, TypeVar, overload
from .token import Token
def _attribute_token(self) -> Token:
    """Return the `Token` that is used as the data source for the
        properties defined below."""
    if self.token:
        return self.token
    if self.nester_tokens:
        return self.nester_tokens.opening
    raise AttributeError('Root node does not have the accessed attribute')