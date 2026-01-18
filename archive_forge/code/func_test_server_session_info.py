import asyncio
import datetime as dt
import logging
import os
import pathlib
import time
import weakref
import param
import pytest
import requests
from bokeh.events import ButtonClick
from panel.config import config
from panel.io import state
from panel.io.resources import DIST_DIR, JS_VERSION
from panel.io.server import INDEX_HTML, get_server, set_curdoc
from panel.layout import Row
from panel.models import HTML as BkHTML
from panel.models.tabulator import TableEditEvent
from panel.pane import Markdown
from panel.param import ParamFunction
from panel.reactive import ReactiveHTML
from panel.template import BootstrapTemplate
from panel.tests.util import serve_and_request, serve_and_wait, wait_until
from panel.widgets import (
def test_server_session_info():
    with config.set(session_history=-1):
        html = Markdown('# Title')
        serve_and_request(html)
        assert state.session_info['total'] == 1
        assert len(state.session_info['sessions']) == 1
        sid, session = list(state.session_info['sessions'].items())[0]
        assert session['user_agent'].startswith('python-requests')
        assert state.session_info['live'] == 0
        doc = list(html._documents.keys())[0]
        session_context = param.Parameterized()
        request = param.Parameterized()
        request.arguments = {}
        session_context.request = request
        session_context._document = doc
        session_context.id = sid
        doc._session_context = weakref.ref(session_context)
        with set_curdoc(doc):
            state._init_session(None)
            assert state.session_info['live'] == 1
    html._server_destroy(session_context)
    state._destroy_session(session_context)
    assert state.session_info['live'] == 0