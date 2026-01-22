import functools
import os
import sys
import collections
import importlib
import warnings
from contextvars import copy_context
from importlib.machinery import ModuleSpec
import pkgutil
import threading
import re
import logging
import time
import mimetypes
import hashlib
import base64
import traceback
from urllib.parse import urlparse
from typing import Dict, Optional, Union
import flask
from importlib_metadata import version as _get_distribution_version
from dash import dcc
from dash import html
from dash import dash_table
from .fingerprint import build_fingerprint, check_fingerprint
from .resources import Scripts, Css
from .dependencies import (
from .development.base_component import ComponentRegistry
from .exceptions import (
from .version import __version__
from ._configs import get_combined_config, pathname_configs, pages_folder_config
from ._utils import (
from . import _callback
from . import _get_paths
from . import _dash_renderer
from . import _validate
from . import _watch
from . import _get_app
from ._grouping import map_grouping, grouping_len, update_args_group
from . import _pages
from ._pages import (
from ._jupyter import jupyter_dash, JupyterDisplayMode
from .types import RendererHooks
def _get_traceback(secret, error: Exception):
    try:
        from werkzeug.debug import tbtools
    except ImportError:
        tbtools = None

    def _get_skip(error):
        from dash._callback import _invoke_callback
        tb = error.__traceback__
        skip = 1
        while tb.tb_next is not None:
            skip += 1
            tb = tb.tb_next
            if tb.tb_frame.f_code is _invoke_callback.__code__:
                return skip
        return skip

    def _do_skip(error):
        from dash._callback import _invoke_callback
        tb = error.__traceback__
        while tb.tb_next is not None:
            if tb.tb_frame.f_code is _invoke_callback.__code__:
                return tb.tb_next
            tb = tb.tb_next
        return error.__traceback__
    if hasattr(tbtools, 'get_current_traceback'):
        return tbtools.get_current_traceback(skip=_get_skip(error)).render_full()
    if hasattr(tbtools, 'DebugTraceback'):
        return tbtools.DebugTraceback(error, skip=_get_skip(error)).render_debugger_html(True, secret, True)
    return ''.join(traceback.format_exception(type(error), error, _do_skip(error)))