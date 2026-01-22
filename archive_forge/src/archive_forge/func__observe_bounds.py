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
@observe('south', 'north', 'east', 'west')
def _observe_bounds(self, change):
    self.set_trait('bounds', ((self.south, self.west), (self.north, self.east)))
    self.set_trait('bounds_polygon', ((self.north, self.west), (self.north, self.east), (self.south, self.east), (self.south, self.west)))