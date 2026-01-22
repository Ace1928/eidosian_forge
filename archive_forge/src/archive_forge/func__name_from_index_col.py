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
def _name_from_index_col(cls, col):
    """
        Get index label.

        Deprecated.

        Parameters
        ----------
        col : str
            Index column.

        Returns
        -------
        str
        """
    if col.startswith(ColNameCodec.IDX_COL_NAME):
        return None
    return col