import asyncio
import os
import re
import signal
import sys
from types import FrameType
from typing import Any, Awaitable, Callable, Optional, Union  # noqa
from gunicorn.config import AccessLogFormat as GunicornAccessLogFormat
from gunicorn.workers import base
from aiohttp import web
from .helpers import set_result
from .web_app import Application
from .web_log import AccessLogger
def _get_valid_log_format(self, source_format: str) -> str:
    if source_format == self.DEFAULT_GUNICORN_LOG_FORMAT:
        return self.DEFAULT_AIOHTTP_LOG_FORMAT
    elif re.search('%\\([^\\)]+\\)', source_format):
        raise ValueError("Gunicorn's style options in form of `%(name)s` are not supported for the log formatting. Please use aiohttp's format specification to configure access log formatting: http://docs.aiohttp.org/en/stable/logging.html#format-specification")
    else:
        return source_format