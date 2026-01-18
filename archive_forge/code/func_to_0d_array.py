from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
from enum import Enum
from pathlib import Path
from typing import (
import numpy as np
import pandas as pd
from xarray.namedarray.utils import (  # noqa: F401
def to_0d_array(value: Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray."""
    if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
        return np.array(value)
    else:
        return to_0d_object_array(value)