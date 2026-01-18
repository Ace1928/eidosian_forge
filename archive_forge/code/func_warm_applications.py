import ast
import base64
import logging
import os
import pathlib
from glob import glob
from types import ModuleType
from bokeh.application import Application
from bokeh.application.handlers.document_lifecycle import (
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.subcommands.serve import Serve as _BkServe
from bokeh.command.util import build_single_handler_applications
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import EMPTY_LAYOUT
from bokeh.server.contexts import ApplicationContext
from tornado.ioloop import PeriodicCallback
from tornado.web import StaticFileHandler
from ..auth import BasicAuthProvider, OAuthProvider
from ..config import config
from ..io.document import _cleanup_doc
from ..io.liveness import LivenessHandler
from ..io.reload import record_modules, watch
from ..io.rest import REST_PROVIDERS
from ..io.server import INDEX_HTML, get_static_routes, set_curdoc
from ..io.state import state
from ..util import fullpath
def warm_applications(self, applications, reuse_sessions):
    from ..io.session import generate_session
    for path, app in applications.items():
        session = generate_session(app)
        with set_curdoc(session.document):
            if config.session_key_func:
                reuse_sessions = False
            else:
                state._session_key_funcs[path] = lambda r: r.path
                state._sessions[path] = session
                session.block_expiration()
            state._on_load(None)
        _cleanup_doc(session.document, destroy=not reuse_sessions)