from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (

            Pop current input context.

            Parameters
            ----------
            type : Any
                An exception type.
            value : Any
                An exception value.
            traceback : Any
                A traceback.
            