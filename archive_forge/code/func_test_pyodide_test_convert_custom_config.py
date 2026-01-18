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
@pytest.mark.parametrize('runtime', ['pyodide', 'pyscript', 'pyodide-worker'])
def test_pyodide_test_convert_custom_config(http_serve, page, runtime):
    wait_for_app(http_serve, config_app, page, runtime)
    assert page.locator('body').evaluate("(element) =>\n        window.getComputedStyle(element).getPropertyValue('background-color')") == 'rgb(0, 0, 255)'