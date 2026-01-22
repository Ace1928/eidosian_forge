import json
import warnings
import os
from plotly import exceptions, optional_imports
from plotly.files import PLOTLY_DIR
def _replace_newline(obj):
    """Replaces '
' with '<br>' for all strings in a collection."""
    if isinstance(obj, dict):
        d = dict()
        for key, val in list(obj.items()):
            d[key] = _replace_newline(val)
        return d
    elif isinstance(obj, list):
        l = list()
        for index, entry in enumerate(obj):
            l += [_replace_newline(entry)]
        return l
    elif isinstance(obj, str):
        s = obj.replace('\n', '<br>')
        if s != obj:
            warnings.warn("Looks like you used a newline character: '\\n'.\n\nPlotly uses a subset of HTML escape characters\nto do things like newline (<br>), bold (<b></b>),\nitalics (<i></i>), etc. Your newline characters \nhave been converted to '<br>' so they will show \nup right on your Plotly figure!")
        return s
    else:
        return obj