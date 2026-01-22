from __future__ import annotations
import asyncio
import random
import inspect
import aiohttpx
import functools
import subprocess
from pydantic import BaseModel
from urllib.parse import urlparse
from lazyops.libs.proxyobj import ProxyObject, proxied
from .base import BaseGlobalClient, cachify
from .utils import aget_root_domain, get_user_agent, http_retry_wrapper
from typing import Optional, Type, TypeVar, Literal, Union, Set, Awaitable, Any, Dict, List, Callable, overload, TYPE_CHECKING
def _get_pdftotext(self, url: str, validate_url: Optional[bool]=False, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, raise_errors: Optional[bool]=None, **kwargs) -> Optional[str]:
    """
        Transform a PDF File to Text directly from URL
        """
    if not self.pdftotext_enabled:
        raise ValueError('pdftotext is not enabled. Please install pdftotext')
    get_func = self.__get_pdftotext
    if retryable:
        get_func = http_retry_wrapper(max_tries=retry_limit + 1)(get_func)
    return get_func(url, raise_errors=raise_errors, **kwargs)