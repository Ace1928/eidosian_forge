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
def _add_unit(self, name, factor, decimals):
    self._custom_units_dict[name] = {'factor': factor, 'display': name, 'decimals': decimals}
    self._custom_units = dict(**self._custom_units_dict)