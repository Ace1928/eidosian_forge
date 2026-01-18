import calendar
import datetime as dt
import re
import time
from collections import defaultdict
from contextlib import contextmanager, suppress
from itertools import permutations
import bokeh
import numpy as np
import pandas as pd
from bokeh.core.json_encoder import serialize_json  # noqa (API import)
from bokeh.core.property.datetime import Datetime
from bokeh.core.validation import silence
from bokeh.layouts import Column, Row, group_tools
from bokeh.models import (
from bokeh.models.formatters import PrintfTickFormatter, TickFormatter
from bokeh.models.scales import CategoricalScale, LinearScale, LogScale
from bokeh.models.widgets import DataTable, Div
from bokeh.plotting import figure
from bokeh.themes import built_in_themes
from bokeh.themes.theme import Theme
from packaging.version import Version
from ...core.layout import Layout
from ...core.ndmapping import NdMapping
from ...core.overlay import NdOverlay, Overlay
from ...core.spaces import DynamicMap, get_nested_dmaps
from ...core.util import (
from ...util.warnings import warn
from ..util import dim_axis_label
def update_shared_sources(f):
    """
    Context manager to ensures data sources shared between multiple
    plots are cleared and updated appropriately avoiding warnings and
    allowing empty frames on subplots. Expects a list of
    shared_sources and a mapping of the columns expected columns for
    each source in the plots handles.
    """

    def wrapper(self, *args, **kwargs):
        source_cols = self.handles.get('source_cols', {})
        shared_sources = self.handles.get('shared_sources', [])
        doc = self.document
        for source in shared_sources:
            source.data.clear()
            if doc:
                event_obj = doc.callbacks
                event_obj._held_events = event_obj._held_events[:-1]
        ret = f(self, *args, **kwargs)
        for source in shared_sources:
            expected = source_cols[id(source)]
            found = [c for c in expected if c in source.data]
            empty = np.full_like(source.data[found[0]], np.nan) if found else []
            patch = {c: empty for c in expected if c not in source.data}
            source.data.update(patch)
        return ret
    return wrapper