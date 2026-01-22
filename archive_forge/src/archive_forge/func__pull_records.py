from __future__ import annotations
from collections import (
import copy
from typing import (
import numpy as np
from pandas._libs.writers import convert_json_to_lines
import pandas as pd
from pandas import DataFrame
def _pull_records(js: dict[str, Any], spec: list | str) -> list:
    """
        Internal function to pull field for records, and similar to
        _pull_field, but require to return list. And will raise error
        if has non iterable value.
        """
    result = _pull_field(js, spec, extract_record=True)
    if not isinstance(result, list):
        if pd.isnull(result):
            result = []
        else:
            raise TypeError(f'{js} has non list value {result} for path {spec}. Must be list or null.')
    return result