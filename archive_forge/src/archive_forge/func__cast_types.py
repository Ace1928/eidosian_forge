from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
import datetime
from enum import Enum
import itertools
from typing import (
import warnings
import numpy as np
from pandas._libs import (
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import (
from pandas.core import algorithms
from pandas.core.arrays import (
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools
from pandas.io.common import is_potential_multi_index
@final
def _cast_types(self, values: ArrayLike, cast_type: DtypeObj, column) -> ArrayLike:
    """
        Cast values to specified type

        Parameters
        ----------
        values : ndarray or ExtensionArray
        cast_type : np.dtype or ExtensionDtype
           dtype to cast values to
        column : string
            column name - used only for error reporting

        Returns
        -------
        converted : ndarray or ExtensionArray
        """
    if isinstance(cast_type, CategoricalDtype):
        known_cats = cast_type.categories is not None
        if not is_object_dtype(values.dtype) and (not known_cats):
            values = lib.ensure_string_array(values, skipna=False, convert_na_value=False)
        cats = Index(values).unique().dropna()
        values = Categorical._from_inferred_categories(cats, cats.get_indexer(values), cast_type, true_values=self.true_values)
    elif isinstance(cast_type, ExtensionDtype):
        array_type = cast_type.construct_array_type()
        try:
            if isinstance(cast_type, BooleanDtype):
                return array_type._from_sequence_of_strings(values, dtype=cast_type, true_values=self.true_values, false_values=self.false_values)
            else:
                return array_type._from_sequence_of_strings(values, dtype=cast_type)
        except NotImplementedError as err:
            raise NotImplementedError(f'Extension Array: {array_type} must implement _from_sequence_of_strings in order to be used in parser methods') from err
    elif isinstance(values, ExtensionArray):
        values = values.astype(cast_type, copy=False)
    elif issubclass(cast_type.type, str):
        values = lib.ensure_string_array(values, skipna=True, convert_na_value=False)
    else:
        try:
            values = astype_array(values, cast_type, copy=True)
        except ValueError as err:
            raise ValueError(f'Unable to convert column {column} to type {cast_type}') from err
    return values