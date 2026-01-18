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
def test_server_thread_pool_change_event(threads):
    button = Button(name='Click')
    button2 = Button(name='Click')
    counts = []

    def cb(event, count=[0]):
        count[0] += 1
        counts.append(count[0])
        time.sleep(0.5)
        count[0] -= 1
    button.on_click(cb)
    button2.on_click(cb)
    layout = Row(button, button2)
    serve_and_request(layout)
    model = list(layout._models.values())[0][0]
    doc = model.document
    with set_curdoc(doc):
        button._server_change(doc, model.ref['id'], None, 'clicks', 0, 1)
        button2._server_change(doc, model.ref['id'], None, 'clicks', 0, 1)
    wait_until(lambda: len(counts) > 0 and max(counts) > 1)