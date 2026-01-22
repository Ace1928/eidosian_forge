from __future__ import annotations
import json as _json
import logging
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPException as HTTPException
from io import BytesIO, IOBase
from ...exceptions import InvalidHeader, TimeoutError
from ...response import BaseHTTPResponse
from ...util.retry import Retry
from .request import EmscriptenRequest
@dataclass
class EmscriptenResponse:
    status_code: int
    headers: dict[str, str]
    body: IOBase | bytes
    request: EmscriptenRequest