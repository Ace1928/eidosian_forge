import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
@default('widget')
def _default_widget(self):
    widget = Output(layout={'height': '40px', 'width': '520px', 'margin': '0px -19px 0px 0px'})
    with widget:
        colormap = self.colormap.scale(self.value_min, self.value_max)
        colormap.caption = self.caption
        display(colormap)
    return widget