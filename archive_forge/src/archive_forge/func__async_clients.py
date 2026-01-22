import datetime
import functools
from io import BytesIO
import ssl
import time
import weakref
from tornado.concurrent import (
from tornado.escape import utf8, native_str
from tornado import gen, httputil
from tornado.ioloop import IOLoop
from tornado.util import Configurable
from typing import Type, Any, Union, Dict, Callable, Optional, cast
@classmethod
def _async_clients(cls) -> Dict[IOLoop, 'AsyncHTTPClient']:
    attr_name = '_async_client_dict_' + cls.__name__
    if not hasattr(cls, attr_name):
        setattr(cls, attr_name, weakref.WeakKeyDictionary())
    return getattr(cls, attr_name)