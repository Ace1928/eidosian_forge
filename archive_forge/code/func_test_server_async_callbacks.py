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
def test_server_async_callbacks():
    button = Button(name='Click')
    counts = []

    async def cb(event, count=[0]):
        count[0] += 1
        counts.append(count[0])
        await asyncio.sleep(1)
        count[0] -= 1
    button.on_click(cb)
    serve_and_request(button)
    doc = list(button._models.values())[0][0].document
    with set_curdoc(doc):
        for _ in range(5):
            button.clicks += 1
    wait_until(lambda: len(counts) > 0 and max(counts) > 1)