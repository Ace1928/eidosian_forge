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
def _check_dims(dim: set[Hashable], all_dims: set[Hashable]) -> None:
    wrong_dims = dim - all_dims - {...}
    if wrong_dims:
        wrong_dims_str = ', '.join((f"'{d!s}'" for d in wrong_dims))
        raise ValueError(f'Dimension(s) {wrong_dims_str} do not exist. Expected one or more of {all_dims}')