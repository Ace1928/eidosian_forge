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
def _group_errors_by_json_path(errors: ValidationErrorList) -> GroupedValidationErrors:
    """Groups errors by the `json_path` attribute of the jsonschema ValidationError
    class. This attribute contains the path to the offending element within
    a chart specification and can therefore be considered as an identifier of an
    'issue' in the chart that needs to be fixed.
    """
    errors_by_json_path = collections.defaultdict(list)
    for err in errors:
        err_key = getattr(err, 'json_path', _json_path(err))
        errors_by_json_path[err_key].append(err)
    return dict(errors_by_json_path)