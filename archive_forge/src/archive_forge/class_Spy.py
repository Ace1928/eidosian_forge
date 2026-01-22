from packaging.version import Version
import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import matplotlib
import numpy as np
import pandas as pd
import panel as pn
import pytest
import xarray as xr
from holoviews.util.transform import dim
from hvplot import bind
from hvplot.interactive import Interactive
from hvplot.tests.util import makeDataFrame, makeMixedDataFrame
from hvplot.xarray import XArrayInteractive
from hvplot.util import bokeh3, param2
class Spy:

    def __init__(self):
        self.count = 0
        self.calls = {}

    def __repr__(self):
        return f'Spy(count={self.count!r}, calls={self.calls!r})'

    def register_call(self, called_args, called_kwargs, **kwargs):
        self.calls[self.count] = CallCtxt(called_args, called_kwargs, **kwargs)
        self.count += 1