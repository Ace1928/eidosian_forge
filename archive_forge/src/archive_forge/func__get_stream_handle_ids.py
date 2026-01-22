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
def _get_stream_handle_ids(self, handles):
    """
        Gather the ids of the plotting handles attached to this callback
        This allows checking that a stream is not given the state
        of a plotting handle it wasn't attached to
        """
    stream_handle_ids = defaultdict(dict)
    for stream in self.streams:
        for h in self.models + self.extra_handles:
            if h in handles:
                handle_id = handles[h].ref['id']
                stream_handle_ids[stream][h] = handle_id
    return stream_handle_ids