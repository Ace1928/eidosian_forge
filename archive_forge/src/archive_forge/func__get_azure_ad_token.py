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
def _get_azure_ad_token(self) -> str | None:
    if self._azure_ad_token is not None:
        return self._azure_ad_token
    provider = self._azure_ad_token_provider
    if provider is not None:
        token = provider()
        if not token or not isinstance(token, str):
            raise ValueError(f'Expected `azure_ad_token_provider` argument to return a string but it returned {token}')
        return token
    return None