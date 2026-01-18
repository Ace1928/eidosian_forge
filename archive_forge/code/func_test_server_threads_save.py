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
def test_server_threads_save(threads, tmp_path):
    button = Button()
    fsave = tmp_path / 'button.html'

    def cb(event):
        button.save(fsave)

    def simulate_click():
        button._comm_event(state.curdoc, ButtonClick(model=None))
    button.on_click(cb)

    def app():
        state.curdoc.add_next_tick_callback(simulate_click)
        return button
    serve_and_request(app)
    wait_until(lambda: fsave.exists())