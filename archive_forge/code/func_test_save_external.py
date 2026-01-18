import re
from io import StringIO
from pathlib import Path
import numpy as np
from bokeh.resources import Resources
from panel.config import config
from panel.io.resources import CDN_DIST
from panel.models.vega import VegaPlot
from panel.pane import Alert, Vega
from panel.tests.util import hv_available
def test_save_external():
    sio = StringIO()
    pane = Vega(vega_example)
    pane.save(sio)
    sio.seek(0)
    html = sio.read()
    for js in VegaPlot.__javascript_raw__:
        assert js.replace(config.npm_cdn, f'{CDN_DIST}bundled/vegaplot') in html