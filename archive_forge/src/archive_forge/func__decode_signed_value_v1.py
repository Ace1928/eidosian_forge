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
def _decode_signed_value_v1(secret: Union[str, bytes], name: str, value: bytes, max_age_days: float, clock: Callable[[], float]) -> Optional[bytes]:
    parts = utf8(value).split(b'|')
    if len(parts) != 3:
        return None
    signature = _create_signature_v1(secret, name, parts[0], parts[1])
    if not hmac.compare_digest(parts[2], signature):
        gen_log.warning('Invalid cookie signature %r', value)
        return None
    timestamp = int(parts[1])
    if timestamp < clock() - max_age_days * 86400:
        gen_log.warning('Expired cookie %r', value)
        return None
    if timestamp > clock() + 31 * 86400:
        gen_log.warning('Cookie timestamp in future; possible tampering %r', value)
        return None
    if parts[1].startswith(b'0'):
        gen_log.warning('Tampered cookie %r', value)
        return None
    try:
        return base64.b64decode(parts[0])
    except Exception:
        return None