from tornado.escape import _unicode
from tornado import gen, version
from tornado.httpclient import (
from tornado import httputil
from tornado.http1connection import HTTP1Connection, HTTP1ConnectionParameters
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError, IOStream
from tornado.netutil import (
from tornado.log import gen_log
from tornado.tcpclient import TCPClient
import base64
import collections
import copy
import functools
import re
import socket
import ssl
import sys
import time
from io import BytesIO
import urllib.parse
from typing import Dict, Any, Callable, Optional, Type, Union
from types import TracebackType
import typing
def _on_end_request(self) -> None:
    self.stream.close()