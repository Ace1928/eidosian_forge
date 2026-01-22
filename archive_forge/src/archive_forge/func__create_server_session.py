from __future__ import annotations
import asyncio
import html
import os
import weakref
from dataclasses import dataclass
from typing import (
import tornado
from bokeh.document import Document
from bokeh.embed.bundle import extension_dirs
from bokeh.protocol import Protocol
from bokeh.protocol.receiver import Receiver
from bokeh.server.connection import ServerConnection
from bokeh.server.contexts import BokehSessionContext
from bokeh.server.protocol_handler import ProtocolHandler
from bokeh.server.session import ServerSession
from bokeh.server.views.static_handler import StaticHandler
from bokeh.server.views.ws import WSHandler
from bokeh.util.token import get_session_id, get_token_payload
from ipykernel.comm import Comm
from ..util import edit_readonly
from .application import build_single_handler_application
from .resources import Resources
from .server import server_html_page_for_session
from .state import set_curdoc, state
def _create_server_session(self) -> ServerSession:
    doc = Document()
    self._context = session_context = BokehSessionContext(self.session_id, None, doc)
    session_context._request = _RequestProxy(arguments={k: [v.encode('utf-8') for v in vs] for k, vs in self.payload.get('arguments', {}).items()}, cookies=self.payload.get('cookies'), headers=self.payload.get('headers'))
    session_context._token = self.token
    doc._session_context = weakref.ref(session_context)
    state._jupyter_kernel_context = session_context
    if self.path.endswith('.yaml') or self.path.endswith('.yml'):
        from lumen.command import build_single_handler_application as build_lumen_app
        app = build_lumen_app(self.path, argv=None)
    else:
        app = build_single_handler_application(self.path)
    with set_curdoc(doc):
        app.initialize_document(doc)
    runner = app._handlers[0]._runner
    loop = tornado.ioloop.IOLoop.current()
    session = JupyterServerSession(self.session_id, doc, io_loop=loop, token=self.token)
    session_context._set_session(session)
    return (session, runner.error_detail)