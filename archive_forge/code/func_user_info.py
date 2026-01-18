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
@property
def user_info(self) -> Dict[str, Any] | None:
    """
        Returns the OAuth user information if enabled.
        """
    is_guest = self.cookies.get('is_guest')
    if is_guest:
        return {'user': 'guest', 'username': 'guest'}
    id_token = self._decode_cookie('id_token')
    if id_token is None:
        return None
    return decode_token(id_token)