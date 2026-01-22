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
def _deep_copy(obj, ignore: Optional[list]=None):
    if ignore is None:
        ignore = []
    if isinstance(obj, SchemaBase):
        args = tuple((_deep_copy(arg) for arg in obj._args))
        kwds = {k: _deep_copy(v, ignore=ignore) if k not in ignore else v for k, v in obj._kwds.items()}
        with debug_mode(False):
            return obj.__class__(*args, **kwds)
    elif isinstance(obj, list):
        return [_deep_copy(v, ignore=ignore) for v in obj]
    elif isinstance(obj, dict):
        return {k: _deep_copy(v, ignore=ignore) if k not in ignore else v for k, v in obj.items()}
    else:
        return obj