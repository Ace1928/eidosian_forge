import sys
from collections import OrderedDict
from IPython.display import display
from ipywidgets import VBox
from ipywidgets import Image as ipyImage
from numpy import arange, issubdtype, array, column_stack, shape
from .figure import Figure
from .scales import Scale, LinearScale, Mercator
from .axes import Axis
from .marks import (Lines, Scatter, ScatterGL, Hist, Bars, OHLC, Pie, Map, Image,
from .toolbar import Toolbar
from .interacts import (BrushIntervalSelector, FastIntervalSelector,
from traitlets.utils.sentinel import Sentinel
import functools
def _data_decorator(func):

    @functools.wraps(func)
    def _mark_with_data(*args, **kwargs):
        data = kwargs.pop('data', None)
        if data is None:
            return func(*args, **kwargs)
        else:
            data_args = [data[i] if hashable(data, i) else i for i in args]
            data_kwargs = {kw: data[kwargs[kw]] if hashable(data, kwargs[kw]) else kwargs[kw] for kw in set(kwarg_names).intersection(list(kwargs.keys()))}
            try:
                data_kwargs['index_data'] = data.index
            except AttributeError:
                pass
            kwargs_update = kwargs.copy()
            kwargs_update.update(data_kwargs)
            return func(*data_args, **kwargs_update)
    return _mark_with_data