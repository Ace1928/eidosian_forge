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
def _destroy_session(self, session_context):
    session_id = session_context.id
    sessions = self.session_info['sessions']
    if session_id in sessions and sessions[session_id]['ended'] is None:
        session = sessions[session_id]
        if session['rendered'] is not None:
            self.session_info['live'] -= 1
        session['ended'] = dt.datetime.now().timestamp()
        self.param.trigger('session_info')
    doc = session_context._document
    if doc in self._periodic:
        for cb in self._periodic[doc]:
            try:
                cb._cleanup(session_context)
            except Exception:
                pass
        del self._periodic[doc]
    if doc in self._locations:
        loc = state._locations[doc]
        loc._server_destroy(session_context)
        del state._locations[doc]
    if doc in self._notifications:
        notification = self._notifications[doc]
        notification._server_destroy(session_context)
        del state._notifications[doc]
    if doc in self._templates:
        del self._templates[doc]