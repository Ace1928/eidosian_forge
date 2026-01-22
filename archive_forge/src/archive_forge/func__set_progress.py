import json
import traceback
from contextvars import copy_context
from _plotly_utils.utils import PlotlyJSONEncoder
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate
from dash.long_callback.managers import BaseLongCallbackManager
def _set_progress(progress_value):
    if not isinstance(progress_value, (list, tuple)):
        progress_value = [progress_value]
    cache.set(progress_key, json.dumps(progress_value, cls=PlotlyJSONEncoder))