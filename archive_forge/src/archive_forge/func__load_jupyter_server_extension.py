from the server to the kernel.
from __future__ import annotations
import asyncio
import calendar
import datetime as dt
import inspect
import json
import logging
import os
import pathlib
import textwrap
import time
from queue import Empty
from typing import Any, Awaitable
from urllib.parse import urljoin
import tornado
from bokeh.embed.bundle import extension_dirs
from bokeh.protocol import Protocol
from bokeh.protocol.exceptions import ProtocolError
from bokeh.protocol.receiver import Receiver
from bokeh.server.tornado import DEFAULT_KEEP_ALIVE_MS
from bokeh.server.views.multi_root_static_handler import MultiRootStaticHandler
from bokeh.server.views.static_handler import StaticHandler
from bokeh.server.views.ws import WSHandler
from bokeh.util.token import (
from jupyter_server.base.handlers import JupyterHandler
from tornado.ioloop import PeriodicCallback
from tornado.web import StaticFileHandler
from ..config import config
from .resources import DIST_DIR, ERROR_TEMPLATE, _env
from .server import COMPONENT_PATH, ComponentResourceHandler
from .state import state
import os
import pathlib
import sys
from panel.io.jupyter_executor import PanelExecutor
def _load_jupyter_server_extension(notebook_app):
    base_url = notebook_app.web_app.settings['base_url']
    notebook_app.web_app.add_handlers(host_pattern='.*$', host_handlers=[(urljoin(base_url, 'panel-preview/static/extensions/(.*)'), MultiRootStaticHandler, dict(root=extension_dirs)), (urljoin(base_url, 'panel-preview/static/(.*)'), StaticHandler), (urljoin(base_url, 'panel-preview/render/(.*)/ws'), PanelWSProxy), (urljoin(base_url, 'panel-preview/render/(.*)'), PanelJupyterHandler, {}), (urljoin(base_url, 'panel-preview/layout/(.*)'), PanelLayoutHandler, {}), (urljoin(base_url, 'panel_dist/(.*)'), StaticFileHandler, dict(path=DIST_DIR)), (urljoin(base_url, f'panel-preview/{COMPONENT_PATH}(.*)'), ComponentResourceHandler, {})])