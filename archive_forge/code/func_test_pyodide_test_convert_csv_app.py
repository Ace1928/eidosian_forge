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
@pytest.mark.parametrize('runtime, http_patch', [('pyodide', False), ('pyodide', True), ('pyodide-worker', False), ('pyodide-worker', True)])
def test_pyodide_test_convert_csv_app(http_serve, page, runtime, http_patch):
    msgs = wait_for_app(http_serve, csv_app, page, runtime, http_patch=http_patch)
    expected_titles = ['index', 'date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']
    titles = page.locator('.tabulator-col-title')
    expect(titles).to_have_count(1 + len(expected_titles), timeout=60 * 1000)
    titles = titles.all_text_contents()
    assert titles[1:] == expected_titles
    assert [msg for msg in msgs if msg.type == 'error' and 'favicon' not in msg.location['url']] == []