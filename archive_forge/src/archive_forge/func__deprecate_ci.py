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
def _deprecate_ci(errorbar, ci):
    """
    Warn on usage of ci= and convert to appropriate errorbar= arg.

    ci was deprecated when errorbar was added in 0.12. It should not be removed
    completely for some time, but it can be moved out of function definitions
    (and extracted from kwargs) after one cycle.

    """
    if ci is not deprecated and ci != 'deprecated':
        if ci is None:
            errorbar = None
        elif ci == 'sd':
            errorbar = 'sd'
        else:
            errorbar = ('ci', ci)
        msg = f'\n\nThe `ci` parameter is deprecated. Use `errorbar={repr(errorbar)}` for the same effect.\n'
        warnings.warn(msg, FutureWarning, stacklevel=3)
    return errorbar