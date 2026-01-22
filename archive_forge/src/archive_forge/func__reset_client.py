from __future__ import annotations
import os as _os
from typing_extensions import override
from . import types
from ._types import NOT_GIVEN, NoneType, NotGiven, Transport, ProxiesTypes
from ._utils import file_from_path
from ._client import Client, OpenAI, Stream, Timeout, Transport, AsyncClient, AsyncOpenAI, AsyncStream, RequestOptions
from ._models import BaseModel
from ._version import __title__, __version__
from ._response import APIResponse as APIResponse, AsyncAPIResponse as AsyncAPIResponse
from ._exceptions import (
from ._utils._logs import setup_logging as _setup_logging
from .lib import azure as _azure
from .version import VERSION as VERSION
from .lib.azure import AzureOpenAI as AzureOpenAI, AsyncAzureOpenAI as AsyncAzureOpenAI
from .lib._old_api import *
from .lib.streaming import (
import typing as _t
import typing_extensions as _te
import httpx as _httpx
from ._base_client import DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES
from ._module_client import (
def _reset_client() -> None:
    global _client
    _client = None