import abc
import time
import random
import asyncio
import contextlib
from pydantic import BaseModel
from lazyops.imports._aiohttpx import resolve_aiohttpx
from lazyops.imports._bs4 import resolve_bs4
from urllib.parse import quote_plus
import aiohttpx
from bs4 import BeautifulSoup, Tag
from .utils import filter_result, load_user_agents, load_cookie_jar, get_random_jitter
from typing import List, Optional, Dict, Any, Union, Tuple, Set, Callable, Awaitable, TypeVar, Generator, AsyncGenerator
def append_extra_params(self, url: str, extra_params: Dict[str, str]) -> str:
    """
        Appends extra params
        """
    if not extra_params:
        return url
    for k, v in extra_params.items():
        k = quote_plus(k)
        v = quote_plus(v)
        url = f'{url}&{k}={v}'
    return url