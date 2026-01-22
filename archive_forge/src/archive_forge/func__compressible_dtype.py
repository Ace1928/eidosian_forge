import os
import re
import sys
import uuid
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping, Sequence
from copy import copy as ccopy
from copy import deepcopy
import datetime
from html import escape
from typing import (
import numpy as np
import xarray as xr
from packaging import version
from ..rcparams import rcParams
from ..utils import HtmlTemplate, _subset_list, _var_names, either_dict_or_kwargs
from .base import _extend_xr_method, _make_json_serializable, dict_to_dataset
def _compressible_dtype(dtype):
    """Check basic dtypes for automatic compression."""
    if dtype.kind == 'V':
        return all((_compressible_dtype(item) for item, _ in dtype.fields.values()))
    return dtype.kind in {'b', 'i', 'u', 'f', 'c', 'S'}