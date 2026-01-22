import base64
import binascii
import datetime
import email.utils
import functools
import gzip
import hashlib
import hmac
import http.cookies
from inspect import isclass
from io import BytesIO
import mimetypes
import numbers
import os.path
import re
import socket
import sys
import threading
import time
import warnings
import tornado
import traceback
import types
import urllib.parse
from urllib.parse import urlencode
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import escape
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado import httputil
from tornado import iostream
from tornado import locale
from tornado.log import access_log, app_log, gen_log
from tornado import template
from tornado.escape import utf8, _unicode
from tornado.routing import (
from tornado.util import ObjectDict, unicode_type, _websocket_mask
from typing import (
from types import TracebackType
import typing
def _decode_fields_v2(value: bytes) -> Tuple[int, bytes, bytes, bytes, bytes]:

    def _consume_field(s: bytes) -> Tuple[bytes, bytes]:
        length, _, rest = s.partition(b':')
        n = int(length)
        field_value = rest[:n]
        if rest[n:n + 1] != b'|':
            raise ValueError('malformed v2 signed value field')
        rest = rest[n + 1:]
        return (field_value, rest)
    rest = value[2:]
    key_version, rest = _consume_field(rest)
    timestamp, rest = _consume_field(rest)
    name_field, rest = _consume_field(rest)
    value_field, passed_sig = _consume_field(rest)
    return (int(key_version), timestamp, name_field, value_field, passed_sig)