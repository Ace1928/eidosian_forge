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
@validate('style_callback')
def _validate_style_callback(self, proposal):
    if not callable(proposal.value):
        raise TraitError('style_callback should be callable (functor/function/lambda)')
    return proposal.value