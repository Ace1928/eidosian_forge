from functools import partial
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from ._base import (
from .utils import (
from ._compat import groupby_apply_include_groups
from ._statistics import EstimateAggregator, WeightedAggregator
from .axisgrid import FacetGrid, _facet_docs
from ._docstrings import DocstringComponents, _core_docs
class _RelationalPlotter(VectorPlotter):
    wide_structure = {'x': '@index', 'y': '@values', 'hue': '@columns', 'style': '@columns'}
    sort = True