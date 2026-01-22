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
def _load_ui_modules(self, modules: Any) -> None:
    if isinstance(modules, types.ModuleType):
        self._load_ui_modules(dict(((n, getattr(modules, n)) for n in dir(modules))))
    elif isinstance(modules, list):
        for m in modules:
            self._load_ui_modules(m)
    else:
        assert isinstance(modules, dict)
        for name, cls in modules.items():
            try:
                if issubclass(cls, UIModule):
                    self.ui_modules[name] = cls
            except TypeError:
                pass