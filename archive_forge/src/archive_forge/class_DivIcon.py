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
class DivIcon(UILayer):
    """DivIcon class.

    Custom lightweight icon for markers that uses a simple <div> element
    instead of an image used for markers.

    Attributes
    ----------
    html : string
        Custom HTML code to put inside the div element,
        empty by default.
    bg_pos : tuple, default [0, 0]
        Optional relative position of the background, in pixels.
    icon_size: tuple, default None
        The size of the icon, in pixels.
    icon_anchor: tuple, default None
        The coordinates of the "tip" of the icon (relative to its top left corner).
        The icon will be aligned so that this point is at the marker's geographical
        location. Centered by default if icon_size is specified.
    popup_anchor: tuple, default None
        The coordinates of the point from which popups will "open", relative to the
        icon anchor.
    """
    _view_name = Unicode('LeafletDivIconView').tag(sync=True)
    _model_name = Unicode('LeafletDivIconModel').tag(sync=True)
    html = Unicode('').tag(sync=True, o=True)
    bg_pos = List([0, 0], allow_none=True).tag(sync=True, o=True)
    icon_size = List(default_value=None, allow_none=True).tag(sync=True, o=True)
    icon_anchor = List(default_value=None, allow_none=True).tag(sync=True, o=True)
    popup_anchor = List([0, 0], allow_none=True).tag(sync=True, o=True)

    @validate('icon_size', 'icon_anchor', 'popup_anchor')
    def _validate_attr(self, proposal):
        value = proposal['value']
        if value is None or len(value) == 0:
            return None
        if len(value) != 2:
            raise TraitError('The value should be of size 2, but {} was given'.format(value))
        return value