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
@validate('icon_size', 'icon_anchor', 'popup_anchor')
def _validate_attr(self, proposal):
    value = proposal['value']
    if value is None or len(value) == 0:
        return None
    if len(value) != 2:
        raise TraitError('The value should be of size 2, but {} was given'.format(value))
    return value