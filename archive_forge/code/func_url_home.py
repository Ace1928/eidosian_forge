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
@property
def url_home(self) -> str:
    """
        Gets the home url
        """
    return f'https://www.google.{self.tld}/'