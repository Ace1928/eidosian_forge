from __future__ import annotations
import warnings
import numpy as np
from packaging.version import parse as parse_version
from dask.utils import derived_from
Implementation of numpy.divide that works with dtype kwarg.

        Temporary compatibility fix for a bug in numpy's version. See
        https://github.com/numpy/numpy/issues/3484 for the relevant issue.