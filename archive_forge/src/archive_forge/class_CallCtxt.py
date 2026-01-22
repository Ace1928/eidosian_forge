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
class CallCtxt:

    def __init__(self, call_args, call_kwargs, **kwargs):
        for k, v in kwargs.items():
            if k in ['args', 'kwargs']:
                raise ValueError("**kwargs passed to CallCtxt can't be named args or kwargs")
            setattr(self, k, v)
        self.args = call_args
        self.kwargs = call_kwargs

    def __repr__(self):
        inner = ''
        for attr in vars(self):
            inner += f'{attr}={getattr(self, attr)!r}, '
        return f'CallCtxt({inner}args={self.args!r}, kwargs={self.kwargs!r})'

    def is_empty(self):
        return not self.args and (not self.kwargs)