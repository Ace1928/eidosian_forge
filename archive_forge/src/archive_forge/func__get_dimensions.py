from __future__ import annotations
import re
import sys
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import lazy_load
from .base import ModelPane
def _get_dimensions(spec, props):
    dimensions = {}
    responsive_height = spec.get('height') == 'container' and props.get('height') is None
    responsive_width = spec.get('width') == 'container' and props.get('width') is None
    if responsive_height and responsive_width:
        dimensions['sizing_mode'] = 'stretch_both'
    elif responsive_width:
        dimensions['sizing_mode'] = 'stretch_width'
    elif responsive_height:
        dimensions['sizing_mode'] = 'stretch_height'
    return dimensions