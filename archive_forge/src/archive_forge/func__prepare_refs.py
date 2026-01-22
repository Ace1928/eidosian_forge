import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
def _prepare_refs(d: Dict[str, Any]) -> Dict[str, Any]:
    """Add _VEGA_LITE_ROOT_URI in front of all $ref values. This function
        recursively iterates through the whole dictionary."""
    for key, value in d.items():
        if key == '$ref':
            d[key] = _VEGA_LITE_ROOT_URI + d[key]
        elif isinstance(value, dict):
            d[key] = _prepare_refs(value)
        elif isinstance(value, list):
            prepared_values = []
            for v in value:
                if isinstance(v, dict):
                    v = _prepare_refs(v)
                prepared_values.append(v)
            d[key] = prepared_values
    return d