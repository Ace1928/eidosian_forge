from __future__ import annotations
from collections import (
import itertools
import numbers
import string
import sys
from typing import (
import numpy as np
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
import pandas as pd
from pandas.api.extensions import (
from pandas.core.indexers import unpack_tuple_and_ellipses
class JSONDtype(ExtensionDtype):
    type = abc.Mapping
    name = 'json'
    na_value: Mapping[str, Any] = UserDict()

    @classmethod
    def construct_array_type(cls) -> type_t[JSONArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return JSONArray