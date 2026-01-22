from __future__ import annotations
import json
import time
import uuid
import email
import asyncio
import inspect
import logging
import platform
import warnings
import email.utils
from types import TracebackType
from random import random
from typing import (
from functools import lru_cache
from typing_extensions import Literal, override, get_origin
import anyio
import httpx
import distro
import pydantic
from httpx import URL, Limits
from pydantic import PrivateAttr
from . import _exceptions
from ._qs import Querystring
from ._files import to_httpx_files, async_to_httpx_files
from ._types import (
from ._utils import is_dict, is_list, is_given, is_mapping
from ._compat import model_copy, model_dump
from ._models import GenericModel, FinalRequestOptions, validate_type, construct_type
from ._response import (
from ._constants import (
from ._streaming import Stream, SSEDecoder, AsyncStream, SSEBytesDecoder
from ._exceptions import (
from ._legacy_response import LegacyAPIResponse
def _process_response_data(self, *, data: object, cast_to: type[ResponseT], response: httpx.Response) -> ResponseT:
    if data is None:
        return cast(ResponseT, None)
    if cast_to is object:
        return cast(ResponseT, data)
    try:
        if inspect.isclass(cast_to) and issubclass(cast_to, ModelBuilderProtocol):
            return cast(ResponseT, cast_to.build(response=response, data=data))
        if self._strict_response_validation:
            return cast(ResponseT, validate_type(type_=cast_to, value=data))
        return cast(ResponseT, construct_type(type_=cast_to, value=data))
    except pydantic.ValidationError as err:
        raise APIResponseValidationError(response=response, body=data) from err