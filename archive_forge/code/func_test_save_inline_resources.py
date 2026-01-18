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
def test_save_inline_resources():
    alert = Alert('# Save test')
    sio = StringIO()
    alert.save(sio, resources='inline')
    sio.seek(0)
    html = sio.read()
    assert 'alert-primary' in html