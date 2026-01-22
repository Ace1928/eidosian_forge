import os
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
from types import ModuleType
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from seaborn._core.typing import deprecated
from seaborn.external.version import Version
from seaborn.external.appdirs import user_cache_dir
def _get_transform_functions(ax, axis):
    """Return the forward and inverse transforms for a given axis."""
    axis_obj = getattr(ax, f'{axis}axis')
    transform = axis_obj.get_transform()
    return (transform.transform, transform.inverted().transform)