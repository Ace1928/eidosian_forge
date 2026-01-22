from __future__ import annotations
import warnings
from collections.abc import Hashable, MutableMapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Union
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
class ObjectVLenStringCoder(VariableCoder):

    def encode(self):
        raise NotImplementedError

    def decode(self, variable: Variable, name: T_Name=None) -> Variable:
        if variable.dtype == object and variable.encoding.get('dtype', False) == str:
            variable = variable.astype(variable.encoding['dtype'])
            return variable
        else:
            return variable