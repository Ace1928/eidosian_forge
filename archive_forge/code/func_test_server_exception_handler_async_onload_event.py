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
@pytest.mark.parametrize('threads, handler', [('threads', synchronous_handler), ('nothreads', synchronous_handler), ('threads', async_handler), ('nothreads', async_handler)])
def test_server_exception_handler_async_onload_event(threads, handler, request):
    request.getfixturevalue(threads)
    exceptions = []

    def exception_handler(e):
        exceptions.append(e)

    def loaded():
        state._schedule_on_load(state.curdoc, None)
    text_input = TextInput()

    def app():
        config.exception_handler = exception_handler
        state.onload(handler)
        state.curdoc.add_next_tick_callback(loaded)
        return text_input
    serve_and_request(app)
    wait_until(lambda: len(exceptions) == 1)