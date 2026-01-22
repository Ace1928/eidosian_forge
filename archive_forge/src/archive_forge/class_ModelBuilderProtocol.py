from __future__ import annotations
from os import PathLike
from typing import (
from typing_extensions import Literal, Protocol, TypeAlias, TypedDict, override, runtime_checkable
import httpx
import pydantic
from httpx import URL, Proxy, Timeout, Response, BaseTransport, AsyncBaseTransport
@runtime_checkable
class ModelBuilderProtocol(Protocol):

    @classmethod
    def build(cls: type[_T], *, response: Response, data: object) -> _T:
        ...