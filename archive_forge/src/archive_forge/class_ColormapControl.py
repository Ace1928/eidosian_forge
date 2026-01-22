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
class ColormapControl(WidgetControl):
    """ColormapControl class, with WidgetControl as parent class.

    A control which contains a colormap.

    Attributes
    ----------
    caption : str, default 'caption'
        The caption of the colormap.
    colormap: branca.colormap.ColorMap instance, default linear.OrRd_06
        The colormap used for the effect.
    value_min : float, default 0.0
        The minimal value taken by the data to be represented by the colormap.
    value_max : float, default 1.0
        The maximal value taken by the data to be represented by the colormap.
    """
    caption = Unicode('caption')
    colormap = Instance(ColorMap, default_value=linear.OrRd_06)
    value_min = CFloat(0.0)
    value_max = CFloat(1.0)

    @default('widget')
    def _default_widget(self):
        widget = Output(layout={'height': '40px', 'width': '520px', 'margin': '0px -19px 0px 0px'})
        with widget:
            colormap = self.colormap.scale(self.value_min, self.value_max)
            colormap.caption = self.caption
            display(colormap)
        return widget