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
def add_legend_element(self, key, value):
    """Add a new legend element.

        Parameters
        ----------
        key: str
            The key for the legend element.
        value: CSS Color
            The value for the legend element.
        """
    self.legend[key] = value
    self.send_state()