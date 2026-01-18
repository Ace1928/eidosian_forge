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
@pytest.mark.xdist_group(name='server')
def test_show_server_info(html_server_session, markdown_server_session):
    *_, html_port = html_server_session
    *_, markdown_port = markdown_server_session
    server_info = repr(state)
    assert f'localhost:{html_port} - HTML' in server_info
    assert f'localhost:{markdown_port} - Markdown' in server_info