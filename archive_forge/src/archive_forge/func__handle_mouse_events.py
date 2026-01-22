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
def _handle_mouse_events(self, _, content, buffers):
    if content.get('event', '') == 'click':
        self._click_callbacks(**content)
    if content.get('event', '') == 'mouseover':
        self._hover_callbacks(**content)