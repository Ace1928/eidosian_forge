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
def _parse_retry_after_header(self, response_headers: Optional[httpx.Headers]=None) -> float | None:
    """Returns a float of the number of seconds (not milliseconds) to wait after retrying, or None if unspecified.

        About the Retry-After header: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After
        See also  https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After#syntax
        """
    if response_headers is None:
        return None
    try:
        retry_ms_header = response_headers.get('retry-after-ms', None)
        return float(retry_ms_header) / 1000
    except (TypeError, ValueError):
        pass
    retry_header = response_headers.get('retry-after')
    try:
        return float(retry_header)
    except (TypeError, ValueError):
        pass
    retry_date_tuple = email.utils.parsedate_tz(retry_header)
    if retry_date_tuple is None:
        return None
    retry_date = email.utils.mktime_tz(retry_date_tuple)
    return float(retry_date - time.time())