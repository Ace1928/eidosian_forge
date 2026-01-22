from __future__ import annotations
import abc
import dataclasses
import json
import os
import re
import stat
import traceback
import uuid
import time
import typing as t
from .http import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .ci import (
from .data import (
@staticmethod
def _create_http_error(response: HttpResponse) -> ApplicationError:
    """Return an exception created from the given HTTP response."""
    response_json = response.json()
    stack_trace = ''
    if 'message' in response_json:
        message = response_json['message']
    elif 'errorMessage' in response_json:
        message = response_json['errorMessage'].strip()
        if 'stackTrace' in response_json:
            traceback_lines = response_json['stackTrace']
            if traceback_lines and isinstance(traceback_lines[0], list):
                traceback_lines = traceback.format_list(traceback_lines)
            trace = '\n'.join([x.rstrip() for x in traceback_lines])
            stack_trace = f'\nTraceback (from remote server):\n{trace}'
    else:
        message = str(response_json)
    return CoreHttpError(response.status_code, message, stack_trace)