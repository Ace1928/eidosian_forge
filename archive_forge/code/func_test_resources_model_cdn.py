import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resources_model_cdn(document):
    resources = Resources(mode='cdn')
    with set_resource_mode('cdn'):
        with set_curdoc(document):
            extension('tabulator')
            assert resources.js_files[:2] == [f'{CDN_DIST}bundled/datatabulator/tabulator-tables@5.5.0/dist/js/tabulator.min.js', f'{CDN_DIST}bundled/datatabulator/luxon/build/global/luxon.min.js']
            assert resources.css_files == [f'{CDN_DIST}bundled/datatabulator/tabulator-tables@5.5.0/dist/css/tabulator_simple.min.css?v={JS_VERSION}']