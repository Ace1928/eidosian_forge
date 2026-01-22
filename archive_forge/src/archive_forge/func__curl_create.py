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
def _curl_create(self) -> pycurl.Curl:
    curl = pycurl.Curl()
    if curl_log.isEnabledFor(logging.DEBUG):
        curl.setopt(pycurl.VERBOSE, 1)
        curl.setopt(pycurl.DEBUGFUNCTION, self._curl_debug)
    if hasattr(pycurl, 'PROTOCOLS'):
        curl.setopt(pycurl.PROTOCOLS, pycurl.PROTO_HTTP | pycurl.PROTO_HTTPS)
        curl.setopt(pycurl.REDIR_PROTOCOLS, pycurl.PROTO_HTTP | pycurl.PROTO_HTTPS)
    return curl