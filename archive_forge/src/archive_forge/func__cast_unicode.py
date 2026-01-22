from __future__ import annotations
import json
import logging
import os
from typing import TYPE_CHECKING, Any
import tornado.websocket
from tornado import gen
from tornado.concurrent import run_on_executor
def _cast_unicode(s: str | bytes) -> str:
    if isinstance(s, bytes):
        return s.decode('utf-8')
    return s