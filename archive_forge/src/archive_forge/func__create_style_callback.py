import asyncio
import base64
import time
from collections import defaultdict
import numpy as np
from bokeh.models import (
from panel.io.state import set_curdoc, state
from ...core.options import CallbackError
from ...core.util import datetime_types, dimension_sanitizer, dt64_to_dt, isequal
from ...element import Table
from ...streams import (
from ...util.warnings import warn
from .util import bokeh33, convert_timestamp
def _create_style_callback(self, cds, glyph):
    stream = self.streams[0]
    col = cds.column_names[0]
    length = len(cds.data[col])
    for style, values in stream.styles.items():
        cds.data[style] = [values[i % len(values)] for i in range(length)]
        setattr(glyph, style, style)
    cb = CustomJS(code=self._style_callback, args={'styles': stream.styles, 'empty': stream.empty_value})
    cds.js_on_change('data', cb)