import asyncio
import socket
import sys
from typing import Callable, List, Optional, Union
from redis.compat import TypedDict
from ..exceptions import ConnectionError, InvalidResponse, RedisError
from ..typing import EncodableT
from ..utils import HIREDIS_AVAILABLE
from .base import AsyncBaseParser, BaseParser
from .socket import (
Async implementation of parser class for connections using Hiredis