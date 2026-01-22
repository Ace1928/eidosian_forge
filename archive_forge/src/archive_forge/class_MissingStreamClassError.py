from __future__ import annotations
import os
import inspect
import logging
import datetime
import functools
from typing import TYPE_CHECKING, Any, Union, Generic, TypeVar, Callable, Iterator, AsyncIterator, cast, overload
from typing_extensions import Awaitable, ParamSpec, override, deprecated, get_origin
import anyio
import httpx
import pydantic
from ._types import NoneType
from ._utils import is_given, extract_type_arg, is_annotated_type
from ._models import BaseModel, is_basemodel
from ._constants import RAW_RESPONSE_HEADER
from ._streaming import Stream, AsyncStream, is_stream_class_type, extract_stream_chunk_type
from ._exceptions import APIResponseValidationError
class MissingStreamClassError(TypeError):

    def __init__(self) -> None:
        super().__init__('The `stream` argument was set to `True` but the `stream_cls` argument was not given. See `openai._streaming` for reference')