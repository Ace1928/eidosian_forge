import sys
from collections.abc import Hashable
from functools import wraps
from packaging.version import Version
from types import FunctionType
import bokeh
import numpy as np
import pandas as pd
import param
import holoviews as hv
def with_hv_extension(func, extension='bokeh', logo=False):
    """If hv.extension is not loaded, load before calling function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if extension and (not getattr(hv.extension, '_loaded', False)):
            from . import hvplot_extension
            hvplot_extension(extension, logo=logo)
        return func(*args, **kwargs)
    return wrapper