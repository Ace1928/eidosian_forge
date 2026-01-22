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
def _calculate_retry_timeout(self, remaining_retries: int, options: FinalRequestOptions, response_headers: Optional[httpx.Headers]=None) -> float:
    max_retries = options.get_max_retries(self.max_retries)
    retry_after = self._parse_retry_after_header(response_headers)
    if retry_after is not None and 0 < retry_after <= 60:
        return retry_after
    nb_retries = max_retries - remaining_retries
    sleep_seconds = min(INITIAL_RETRY_DELAY * pow(2.0, nb_retries), MAX_RETRY_DELAY)
    jitter = 1 - 0.25 * random()
    timeout = sleep_seconds * jitter
    return timeout if timeout >= 0 else 0