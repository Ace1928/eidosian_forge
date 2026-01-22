from __future__ import annotations
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import get_string_dtype, pyarrow_strings_enabled
from dask.utils import maybe_pluralize
def _format_footer(suffix='', layers=1):
    if pyarrow_strings_enabled():
        return f'Dask Name: to_pyarrow_string{suffix}, {maybe_pluralize(layers + 1, 'graph layer')}'
    return f'Dask Name: from_pandas{suffix}, {maybe_pluralize(layers, 'graph layer')}'