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
def _handle_request_exception(self, e: BaseException) -> None:
    if isinstance(e, Finish):
        if not self._finished:
            self.finish(*e.args)
        return
    try:
        self.log_exception(*sys.exc_info())
    except Exception:
        app_log.error('Error in exception logger', exc_info=True)
    if self._finished:
        return
    if isinstance(e, HTTPError):
        self.send_error(e.status_code, exc_info=sys.exc_info())
    else:
        self.send_error(500, exc_info=sys.exc_info())