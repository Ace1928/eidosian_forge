import copy
from textwrap import dedent
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import utils
from . import algorithms as algo
from .axisgrid import FacetGrid, _facet_docs
def _check_statsmodels(self):
    """Check whether statsmodels is installed if any boolean options require it."""
    options = ('logistic', 'robust', 'lowess')
    err = '`{}=True` requires statsmodels, an optional dependency, to be installed.'
    for option in options:
        if getattr(self, option) and (not _has_statsmodels):
            raise RuntimeError(err.format(option))