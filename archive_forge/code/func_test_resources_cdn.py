import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resources_cdn():
    resources = Resources(mode='cdn', minified=True)
    assert resources.js_raw == ['Bokeh.set_log_level("info");']
    assert resources.js_files == [f'https://cdn.bokeh.org/bokeh/{bk_prefix}/bokeh-{bokeh_version}.min.js', f'https://cdn.bokeh.org/bokeh/{bk_prefix}/bokeh-gl-{bokeh_version}.min.js', f'https://cdn.bokeh.org/bokeh/{bk_prefix}/bokeh-widgets-{bokeh_version}.min.js', f'https://cdn.bokeh.org/bokeh/{bk_prefix}/bokeh-tables-{bokeh_version}.min.js', f'https://cdn.bokeh.org/bokeh/{bk_prefix}/bokeh-mathjax-{bokeh_version}.min.js']