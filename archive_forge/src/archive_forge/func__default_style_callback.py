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
@default('style_callback')
def _default_style_callback(self):

    def compute_style(feature, colormap, choro_data):
        return dict(fillColor=self.nan_color if isnan(choro_data) else colormap(choro_data), fillOpacity=self.nan_opacity if isnan(choro_data) else self.default_opacity, color='black', weight=0.9)
    return compute_style