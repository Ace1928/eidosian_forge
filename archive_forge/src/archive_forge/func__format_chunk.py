import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
def _format_chunk(self, chunk: bytes) -> bytes:
    if self._expected_content_remaining is not None:
        self._expected_content_remaining -= len(chunk)
        if self._expected_content_remaining < 0:
            self.stream.close()
            raise httputil.HTTPOutputError('Tried to write more data than Content-Length')
    if self._chunking_output and chunk:
        return utf8('%x' % len(chunk)) + b'\r\n' + chunk + b'\r\n'
    else:
        return chunk