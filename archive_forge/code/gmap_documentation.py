from __future__ import annotations
import logging # isort:skip
from ..models import (
from ._figure import BaseFigureOptions
from ._plot import _get_num_minor_ticks
from ._tools import process_active_tools, process_tools_arg
from .glyph_api import GlyphAPI
 Create a new :class:`~bokeh.plotting.GMap` for plotting.

    Args:
        google_api_key (str):
            Google requires an API key be supplied for maps to function. See:

            https://developers.google.com/maps/documentation/javascript/get-api-key

            The Google API key will be stored as a base64-encoded string in
            the Bokeh Document JSON.

        map_options: (:class:`~bokeh.models.map_plots.GMapOptions`)
            Configuration specific to a Google Map

    All other keyword arguments are passed to :class:`~bokeh.plotting.GMap`.

    Returns:
       :class:`~bokeh.plotting.GMap`

    