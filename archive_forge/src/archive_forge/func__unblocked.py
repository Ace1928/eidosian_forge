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
def _unblocked(self, doc: Document) -> bool:
    """
        Indicates whether Document events can be dispatched or have
        to scheduled on the event loop. Events can only be safely
        dispatched if:

        1. The Document to be modified is the same one that the server
           is currently processing.
        2. We are on the same thread that the Document was created on.
        3. The application has fully loaded and the Websocket is open.
        """
    return doc is self.curdoc and self._thread_id in (self._current_thread, None) and (not doc or not doc.session_context or self._loaded.get(doc))