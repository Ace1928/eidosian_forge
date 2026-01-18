import concurrent.futures
from io import BytesIO
import tornado
import sys
from tornado.concurrent import dummy_executor
from tornado import escape
from tornado import httputil
from tornado.ioloop import IOLoop
from tornado.log import access_log
from typing import List, Tuple, Optional, Callable, Any, Dict, Text
from types import TracebackType
import typing
def to_wsgi_str(s: bytes) -> str:
    assert isinstance(s, bytes)
    return s.decode('latin1')