from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
def _legend_add_subtitle(handles, labels, text):
    """Add a subtitle to legend handles."""
    import matplotlib.pyplot as plt
    if text and len(handles) > 1:
        blank_handle = plt.Line2D([], [], label=text)
        blank_handle.set_visible(False)
        handles = [blank_handle] + handles
        labels = [text] + labels
    return (handles, labels)