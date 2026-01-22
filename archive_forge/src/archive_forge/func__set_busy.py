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
def _set_busy(self, busy):
    """
        Sets panel.state to busy if available.
        """
    if 'busy' not in state.param:
        return
    from panel.util import edit_readonly
    with edit_readonly(state):
        state.busy = busy