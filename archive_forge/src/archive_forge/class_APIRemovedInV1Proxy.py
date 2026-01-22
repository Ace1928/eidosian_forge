from __future__ import annotations
from typing import TYPE_CHECKING, Any
from typing_extensions import override
from .._utils import LazyProxy
from .._exceptions import OpenAIError
class APIRemovedInV1Proxy(LazyProxy[Any]):

    def __init__(self, *, symbol: str) -> None:
        super().__init__()
        self._symbol = symbol

    @override
    def __load__(self) -> Any:
        return self

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise APIRemovedInV1(symbol=self._symbol)