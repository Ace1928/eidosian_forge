from __future__ import annotations
import asyncio
import dataclasses
import datetime as dt
import gc
import inspect
import json
import logging
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from functools import partial, wraps
from typing import (
from bokeh.application.application import SessionContext
from bokeh.core.serialization import Serializable
from bokeh.document.document import Document
from bokeh.document.events import (
from bokeh.model.util import visit_immediate_value_references
from bokeh.models import CustomJS
from ..config import config
from ..util import param_watchers
from .loading import LOADING_INDICATOR_CSS_CLASS
from .model import hold, monkeypatch_events  # noqa: F401 API import
from .state import curdoc_locked, state
def _cleanup_doc(doc, destroy=True):
    for callback in doc.session_destroyed_callbacks:
        try:
            callback(None)
        except Exception:
            pass
    if hasattr(doc.callbacks, '_change_callbacks'):
        doc.callbacks._change_callbacks[None] = {}
    from ..viewable import Viewable
    views = {}
    for ref, (pane, root, vdoc, comm) in list(state._views.items()):
        if vdoc is doc:
            pane._cleanup(root)
            if isinstance(pane, Viewable):
                pane._hooks = []
                for p in pane.select():
                    p._hooks = []
                    param_watchers(p, {})
                    p._documents = {}
                    p._internal_callbacks = {}
            param_watchers(pane, {})
            pane._documents = {}
            pane._internal_callbacks = {}
        else:
            views[ref] = (pane, root, doc, comm)
    state._views = views
    if not destroy:
        return
    if doc in state._templates:
        tmpl = state._templates[doc]
        tmpl._documents = {}
        del state._templates[doc]
    doc.destroy(None)