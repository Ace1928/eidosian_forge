import re
import numpy as np
import pandas
import pyarrow
from modin.config import DoUseCalcite
from modin.core.dataframe.pandas.partitioning.partition_manager import (
from modin.error_message import ErrorMessage
from modin.pandas.utils import is_scalar
from ..calcite_builder import CalciteBuilder
from ..calcite_serializer import CalciteSerializer
from ..dataframe.utils import ColNameCodec, is_supported_arrow_type
from ..db_worker import DbTable, DbWorker
from ..partitioning.partition import HdkOnNativeDataframePartition
@classmethod
def _names_from_index_cols(cls, cols):
    """
        Get index labels.

        Deprecated.

        Parameters
        ----------
        cols : list of str
            Index columns.

        Returns
        -------
        list of str
        """
    if len(cols) == 1:
        return cls._name_from_index_col(cols[0])
    return [cls._name_from_index_col(n) for n in cols]