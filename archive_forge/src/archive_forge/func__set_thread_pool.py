import ast
import copy
import importlib
import inspect
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from weakref import WeakKeyDictionary
import param
from bokeh.core.has_props import _default_resolver
from bokeh.document import Document
from bokeh.model import Model
from bokeh.settings import settings as bk_settings
from pyviz_comms import (
from .io.logging import panel_log_handler
from .io.state import state
from .util import param_watchers
@param.depends('_nthreads', watch=True, on_init=True)
def _set_thread_pool(self):
    if self.nthreads is None:
        if state._thread_pool is not None:
            state._thread_pool.shutdown(wait=False)
        state._thread_pool = None
        return
    if state._thread_pool:
        raise RuntimeError('Thread pool already running')
    threads = self.nthreads if self.nthreads else None
    state._thread_pool = ThreadPoolExecutor(max_workers=threads)