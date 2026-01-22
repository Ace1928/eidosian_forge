from __future__ import annotations
from typing import TYPE_CHECKING, Any
from typing_extensions import override
from .._utils import LazyProxy
from .._exceptions import OpenAIError
class APIRemovedInV1(OpenAIError):

    def __init__(self, *, symbol: str) -> None:
        super().__init__(INSTRUCTIONS.format(symbol=symbol))