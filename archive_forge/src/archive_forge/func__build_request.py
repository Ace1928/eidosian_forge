from __future__ import annotations
import os
import inspect
from typing import Any, Union, Mapping, TypeVar, Callable, Awaitable, overload
from typing_extensions import Self, override
import httpx
from .._types import NOT_GIVEN, Omit, Timeout, NotGiven
from .._utils import is_given, is_mapping
from .._client import OpenAI, AsyncOpenAI
from .._models import FinalRequestOptions
from .._streaming import Stream, AsyncStream
from .._exceptions import OpenAIError
from .._base_client import DEFAULT_MAX_RETRIES, BaseClient
@override
def _build_request(self, options: FinalRequestOptions) -> httpx.Request:
    if options.url in _deployments_endpoints and is_mapping(options.json_data):
        model = options.json_data.get('model')
        if model is not None and (not '/deployments' in str(self.base_url)):
            options.url = f'/deployments/{model}{options.url}'
    return super()._build_request(options)