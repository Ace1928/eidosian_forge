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
class SelectionXYCallback(BoundsCallback):
    """
    Converts a bounds selection to numeric or categorical x-range
    and y-range selections.
    """

    def _process_msg(self, msg):
        msg = super()._process_msg(msg)
        if 'bounds' not in msg:
            return msg
        el = self.plot.current_frame
        x0, y0, x1, y1 = msg['bounds']
        x_range = self.plot.handles['x_range']
        if isinstance(x_range, FactorRange):
            x0, x1 = (int(round(x0)), int(round(x1)))
            xfactors = x_range.factors[x0:x1]
            if x_range.tags and x_range.tags[0]:
                xdim = el.get_dimension(x_range.tags[0][0][0])
                if xdim and hasattr(el, 'interface'):
                    dtype = el.interface.dtype(el, xdim)
                    try:
                        xfactors = list(np.array(xfactors).astype(dtype))
                    except Exception:
                        pass
            msg['x_selection'] = xfactors
        else:
            msg['x_selection'] = (x0, x1)
        y_range = self.plot.handles['y_range']
        if isinstance(y_range, FactorRange):
            y0, y1 = (int(round(y0)), int(round(y1)))
            yfactors = y_range.factors[y0:y1]
            if y_range.tags and y_range.tags[0]:
                ydim = el.get_dimension(y_range.tags[0][0][0])
                if ydim and hasattr(el, 'interface'):
                    dtype = el.interface.dtype(el, ydim)
                    try:
                        yfactors = list(np.array(yfactors).astype(dtype))
                    except Exception:
                        pass
            msg['y_selection'] = yfactors
        else:
            msg['y_selection'] = (y0, y1)
        return msg