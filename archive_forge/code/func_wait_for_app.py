import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import uuid
from subprocess import PIPE, Popen
import pytest
from playwright.sync_api import expect
from panel.config import config
from panel.io.convert import BOKEH_LOCAL_WHL, PANEL_LOCAL_WHL, convert_apps
import panel as pn
import panel as pn
import panel as pn
import panel as pn
import pandas as pd
import pandas as pd
import sys
import panel as pn
import panel as pn
import panel as pn
import panel as pn
def wait_for_app(http_serve, app, page, runtime, wait=True, **kwargs):
    app_path = http_serve(app)
    convert_apps([app_path], app_path.parent, runtime=runtime, build_pwa=False, prerender=False, panel_version='local', inline=True, **kwargs)
    msgs = []
    page.on('console', lambda msg: msgs.append(msg))
    page.goto(f'http://127.0.0.1:{HTTP_PORT}/{app_path.name[:-3]}.html')
    cls = f'pn-loading pn-{config.loading_spinner}'
    expect(page.locator('body')).to_have_class(cls)
    if wait:
        expect(page.locator('body')).not_to_have_class(cls, timeout=TIMEOUT)
    return msgs