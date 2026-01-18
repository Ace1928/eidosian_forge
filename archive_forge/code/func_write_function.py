import collections
import functools
import logging
import pycurl
import threading
import time
from io import BytesIO
from tornado import httputil
from tornado import ioloop
from tornado.escape import utf8, native_str
from tornado.httpclient import (
from tornado.log import app_log
from typing import Dict, Any, Callable, Union, Optional
import typing
def write_function(b: Union[bytes, bytearray]) -> int:
    assert request.streaming_callback is not None
    self.io_loop.add_callback(request.streaming_callback, b)
    return len(b)