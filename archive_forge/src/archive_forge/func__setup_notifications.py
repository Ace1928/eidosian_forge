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
@param.depends('notifications', watch=True)
def _setup_notifications(self):
    from .io.notifications import NotificationArea
    from .reactive import ReactiveHTMLMetaclass
    if self.notifications and 'notifications' not in ReactiveHTMLMetaclass._loaded_extensions:
        ReactiveHTMLMetaclass._loaded_extensions.add('notifications')
    if not state.curdoc:
        state._notification = NotificationArea()