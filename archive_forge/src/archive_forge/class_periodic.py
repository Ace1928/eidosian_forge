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
class periodic:
    """
    Mocks the API of periodic Thread in hv.core.util, allowing a smooth
    API transition on bokeh server.
    """

    def __init__(self, document):
        self.document = document
        self.callback = None
        self.period = None
        self.count = None
        self.counter = None
        self._start_time = None
        self.timeout = None
        self._pcb = None

    @property
    def completed(self):
        return self.counter is None

    def start(self):
        self._start_time = time.time()
        if self.document is None:
            raise RuntimeError('periodic was registered to be run on bokehserver but no document was found.')
        self._pcb = self.document.add_periodic_callback(self._periodic_callback, self.period)

    def __call__(self, period, count, callback, timeout=None, block=False):
        if isinstance(count, int):
            if count < 0:
                raise ValueError('Count value must be positive')
        elif type(count) is not type(None):
            raise ValueError('Count value must be a positive integer or None')
        self.callback = callback
        self.period = period * 1000.0
        self.timeout = timeout
        self.count = count
        self.counter = 0
        return self

    def _periodic_callback(self):
        self.callback(self.counter)
        self.counter += 1
        if self.timeout is not None:
            dt = time.time() - self._start_time
            if dt > self.timeout:
                self.stop()
        if self.counter == self.count:
            self.stop()

    def stop(self):
        self.counter = None
        self.timeout = None
        try:
            self.document.remove_periodic_callback(self._pcb)
        except ValueError:
            pass
        self._pcb = None

    def __repr__(self):
        return f'periodic({self.period}, {self.count}, {callable_name(self.callback)})'

    def __str__(self):
        return repr(self)