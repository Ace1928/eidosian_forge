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
class SelectModeCallback(Callback):
    attributes = {'box_mode': 'box_select.mode', 'lasso_mode': 'lasso_select.mode'}
    models = ['box_select', 'lasso_select']
    on_changes = ['mode']

    def _process_msg(self, msg):
        stream = self.streams[0]
        if 'box_mode' in msg:
            mode = msg.pop('box_mode')
            if mode != stream.mode:
                msg['mode'] = mode
        if 'lasso_mode' in msg:
            mode = msg.pop('lasso_mode')
            if mode != stream.mode:
                msg['mode'] = mode
        return msg