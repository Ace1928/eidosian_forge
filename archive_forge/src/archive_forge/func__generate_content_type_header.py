import asyncio
import collections.abc
import datetime
import enum
import json
import math
import time
import warnings
from concurrent.futures import Executor
from http import HTTPStatus
from http.cookies import SimpleCookie
from typing import (
from multidict import CIMultiDict, istr
from . import hdrs, payload
from .abc import AbstractStreamWriter
from .compression_utils import ZLibCompressor
from .helpers import (
from .http import SERVER_SOFTWARE, HttpVersion10, HttpVersion11
from .payload import Payload
from .typedefs import JSONEncoder, LooseHeaders
def _generate_content_type_header(self, CONTENT_TYPE: istr=hdrs.CONTENT_TYPE) -> None:
    assert self._content_dict is not None
    assert self._content_type is not None
    params = '; '.join((f'{k}={v}' for k, v in self._content_dict.items()))
    if params:
        ctype = self._content_type + '; ' + params
    else:
        ctype = self._content_type
    self._headers[CONTENT_TYPE] = ctype