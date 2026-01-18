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
def test_server_component_custom_resources_with_prefix_relative_url(port):
    component = CustomComponent()
    r = serve_and_request({'component': component}, prefix='/prefix', suffix='/prefix/component')
    assert f'href="components/panel.tests.test_server/CustomComponent/__css__/assets/custom.css?v={JS_VERSION}"' in r.content.decode('utf-8')