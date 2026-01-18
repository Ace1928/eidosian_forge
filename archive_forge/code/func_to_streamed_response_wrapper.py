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
def to_streamed_response_wrapper(func: Callable[P, R]) -> Callable[P, ResponseContextManager[APIResponse[R]]]:
    """Higher order function that takes one of our bound API methods and wraps it
    to support streaming and returning the raw `APIResponse` object directly.
    """

    @functools.wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> ResponseContextManager[APIResponse[R]]:
        extra_headers: dict[str, str] = {**(cast(Any, kwargs.get('extra_headers')) or {})}
        extra_headers[RAW_RESPONSE_HEADER] = 'stream'
        kwargs['extra_headers'] = extra_headers
        make_request = functools.partial(func, *args, **kwargs)
        return ResponseContextManager(cast(Callable[[], APIResponse[R]], make_request))
    return wrapped