from __future__ import annotations
import json
import sys
from collections import defaultdict
from typing import (
import numpy as np
import param
from bokeh.core.serialization import Serializer
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import is_dataframe, lazy_load
from .base import ModelPane

    The `DeckGL` pane renders the Deck.gl
    JSON specification as well as PyDeck plots inside a panel.

    Deck.gl is a very powerful WebGL-powered framework for visual exploratory
    data analysis of large datasets.

    Reference: https://panel.holoviz.org/reference/panes/DeckGL.html

    :Example:

    >>> pn.extension('deckgl')
    >>> DeckGL(
    ...    some_deckgl_dict_or_pydeck_object,
    ...    mapbox_api_key=MAPBOX_KEY, height=600
    ... )
    