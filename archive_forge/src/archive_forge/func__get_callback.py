from __future__ import annotations
import asyncio
import datetime as dt
import inspect
import logging
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from typing import (
from urllib.parse import urljoin
from weakref import WeakKeyDictionary
import param
from bokeh.document import Document
from bokeh.document.locking import UnlockedDocumentProxy
from bokeh.io import curdoc as _curdoc
from pyviz_comms import CommManager as _CommManager
from ..util import decode_token, parse_timedelta
from .logging import LOG_SESSION_RENDERED, LOG_USER_MSG
def _get_callback(self, endpoint: str):
    _updating: Dict[int, bool] = {}

    def link(*events):
        event = events[0]
        obj = event.cls if event.obj is None else event.obj
        parameterizeds = self._rest_endpoints[endpoint][0]
        if obj not in parameterizeds:
            return
        updating = _updating.get(id(obj), [])
        values = {event.name: event.new for event in events if event.name not in updating}
        if not values:
            return
        _updating[id(obj)] = list(values)
        for parameterized in parameterizeds:
            if id(parameterized) in _updating:
                continue
            try:
                parameterized.param.update(**values)
            except Exception:
                raise
            finally:
                if id(obj) in _updating:
                    not_updated = [p for p in _updating[id(obj)] if p not in values]
                    if not_updated:
                        _updating[id(obj)] = not_updated
                    else:
                        del _updating[id(obj)]
    return link