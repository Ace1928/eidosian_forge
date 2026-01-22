from __future__ import annotations
import asyncio
import datetime as dt
import importlib
import inspect
import logging
import os
import pathlib
import signal
import sys
import threading
import uuid
from contextlib import contextmanager
from functools import partial, wraps
from html import escape
from types import FunctionType, MethodType
from typing import (
from urllib.parse import urljoin, urlparse
import bokeh
import param
import tornado
from bokeh.application import Application as BkApplication
from bokeh.application.handlers.function import FunctionHandler
from bokeh.core.json_encoder import serialize_json
from bokeh.core.templates import AUTOLOAD_JS, FILE, MACROS
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import EMPTY_LAYOUT
from bokeh.embed.bundle import Script
from bokeh.embed.elements import script_for_render_items
from bokeh.embed.util import RenderItem
from bokeh.embed.wrappers import wrap_in_script_tag
from bokeh.models import CustomJS
from bokeh.server.server import Server as BokehServer
from bokeh.server.urls import per_app_patterns, toplevel_patterns
from bokeh.server.views.autoload_js_handler import (
from bokeh.server.views.doc_handler import DocHandler as BkDocHandler
from bokeh.server.views.root_handler import RootHandler as BkRootHandler
from bokeh.server.views.static_handler import StaticHandler
from bokeh.util.serialization import make_id
from bokeh.util.token import (
from tornado.ioloop import IOLoop
from tornado.web import (
from tornado.wsgi import WSGIContainer
from ..config import config
from ..util import edit_readonly, fullpath
from ..util.warnings import warn
from .application import Application, build_single_handler_application
from .document import (  # noqa
from .liveness import LivenessHandler
from .loading import LOADING_INDICATOR_CSS_CLASS
from .logging import (
from .reload import record_modules
from .resources import (
from .session import generate_session
from .state import set_curdoc, state
class ProxyFallbackHandler(RequestHandler):
    """A `RequestHandler` that wraps another HTTP server callback and
    proxies the subpath.
    """

    def initialize(self, fallback, proxy=None):
        self.fallback = fallback
        self.proxy = proxy

    def prepare(self):
        if self.proxy:
            self.request.path = self.request.path.replace(self.proxy, '')
        self.fallback(self.request)
        self._finished = True
        self.on_finish()