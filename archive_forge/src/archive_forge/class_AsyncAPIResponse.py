from __future__ import annotations
import os
import inspect
import logging
import datetime
import functools
from types import TracebackType
from typing import (
from typing_extensions import Awaitable, ParamSpec, override, get_origin
import anyio
import httpx
import pydantic
from ._types import NoneType
from ._utils import is_given, extract_type_arg, is_annotated_type, extract_type_var_from_base
from ._models import BaseModel, is_basemodel
from ._constants import RAW_RESPONSE_HEADER, OVERRIDE_CAST_TO_HEADER
from ._streaming import Stream, AsyncStream, is_stream_class_type, extract_stream_chunk_type
from ._exceptions import OpenAIError, APIResponseValidationError
class AsyncAPIResponse(BaseAPIResponse[R]):

    @overload
    async def parse(self, *, to: type[_T]) -> _T:
        ...

    @overload
    async def parse(self) -> R:
        ...

    async def parse(self, *, to: type[_T] | None=None) -> R | _T:
        """Returns the rich python representation of this response's data.

        For lower-level control, see `.read()`, `.json()`, `.iter_bytes()`.

        You can customise the type that the response is parsed into through
        the `to` argument, e.g.

        ```py
        from openai import BaseModel


        class MyModel(BaseModel):
            foo: str


        obj = response.parse(to=MyModel)
        print(obj.foo)
        ```

        We support parsing:
          - `BaseModel`
          - `dict`
          - `list`
          - `Union`
          - `str`
          - `httpx.Response`
        """
        cache_key = to if to is not None else self._cast_to
        cached = self._parsed_by_type.get(cache_key)
        if cached is not None:
            return cached
        if not self._is_sse_stream:
            await self.read()
        parsed = self._parse(to=to)
        if is_given(self._options.post_parser):
            parsed = self._options.post_parser(parsed)
        self._parsed_by_type[cache_key] = parsed
        return parsed

    async def read(self) -> bytes:
        """Read and return the binary response content."""
        try:
            return await self.http_response.aread()
        except httpx.StreamConsumed as exc:
            raise StreamAlreadyConsumed() from exc

    async def text(self) -> str:
        """Read and decode the response content into a string."""
        await self.read()
        return self.http_response.text

    async def json(self) -> object:
        """Read and decode the JSON response content."""
        await self.read()
        return self.http_response.json()

    async def close(self) -> None:
        """Close the response and release the connection.

        Automatically called if the response body is read to completion.
        """
        await self.http_response.aclose()

    async def iter_bytes(self, chunk_size: int | None=None) -> AsyncIterator[bytes]:
        """
        A byte-iterator over the decoded response content.

        This automatically handles gzip, deflate and brotli encoded responses.
        """
        async for chunk in self.http_response.aiter_bytes(chunk_size):
            yield chunk

    async def iter_text(self, chunk_size: int | None=None) -> AsyncIterator[str]:
        """A str-iterator over the decoded response content
        that handles both gzip, deflate, etc but also detects the content's
        string encoding.
        """
        async for chunk in self.http_response.aiter_text(chunk_size):
            yield chunk

    async def iter_lines(self) -> AsyncIterator[str]:
        """Like `iter_text()` but will only yield chunks for each line"""
        async for chunk in self.http_response.aiter_lines():
            yield chunk