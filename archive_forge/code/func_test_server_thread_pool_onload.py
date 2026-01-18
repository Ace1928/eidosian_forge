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
def test_server_thread_pool_onload(threads):
    counts = []

    def app(count=[0]):
        button = Button(name='Click')

        def onload():
            count[0] += 1
            counts.append(count[0])
            time.sleep(2)
            count[0] -= 1
        state.onload(onload)

        def loaded():
            state._schedule_on_load(state.curdoc, None)
        state.execute(loaded, schedule=True)
        return button
    serve_and_request(app, n=2)
    wait_until(lambda: len(counts) > 0 and max(counts) > 1)