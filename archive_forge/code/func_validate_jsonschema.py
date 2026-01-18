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
def validate_jsonschema(spec, schema, rootschema=None, *, raise_error=True):
    """Validates the passed in spec against the schema in the context of the
    rootschema. If any errors are found, they are deduplicated and prioritized
    and only the most relevant errors are kept. Errors are then either raised
    or returned, depending on the value of `raise_error`.
    """
    errors = _get_errors_from_spec(spec, schema, rootschema=rootschema)
    if errors:
        leaf_errors = _get_leaves_of_error_tree(errors)
        grouped_errors = _group_errors_by_json_path(leaf_errors)
        grouped_errors = _subset_to_most_specific_json_paths(grouped_errors)
        grouped_errors = _deduplicate_errors(grouped_errors)
        main_error = list(grouped_errors.values())[0][0]
        main_error._all_errors = grouped_errors
        if raise_error:
            raise main_error
        else:
            return main_error
    else:
        return None