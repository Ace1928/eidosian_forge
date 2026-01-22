from __future__ import annotations
import json
from enum import Enum
from typing import TYPE_CHECKING, Dict, Final, Literal, Mapping, Union
from typing_extensions import TypeAlias
from streamlit.elements.lib.column_types import ColumnConfig, ColumnType
from streamlit.elements.lib.dicttools import remove_none_values
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.type_util import DataFormat, is_colum_type_arrow_incompatible
def apply_data_specific_configs(columns_config: ColumnConfigMapping, data_df: DataFrame, data_format: DataFormat, check_arrow_compatibility: bool=False) -> None:
    """Apply data specific configurations to the provided dataframe.

    This will apply inplace changes to the dataframe and the column configurations
    depending on the data format.

    Parameters
    ----------
    columns_config : ColumnConfigMapping
        A mapping of column names/ids to column configurations.

    data_df : pd.DataFrame
        The dataframe to apply the configurations to.

    data_format : DataFormat
        The format of the data.

    check_arrow_compatibility : bool
        Whether to check if the data is compatible with arrow.
    """
    import pandas as pd
    if check_arrow_compatibility:
        for column_name, column_data in data_df.items():
            if is_colum_type_arrow_incompatible(column_data):
                update_column_config(columns_config, column_name, {'disabled': True})
                data_df[column_name] = column_data.astype('string')
    if data_format in [DataFormat.SET_OF_VALUES, DataFormat.TUPLE_OF_VALUES, DataFormat.LIST_OF_VALUES, DataFormat.NUMPY_LIST, DataFormat.NUMPY_MATRIX, DataFormat.LIST_OF_RECORDS, DataFormat.LIST_OF_ROWS, DataFormat.COLUMN_VALUE_MAPPING]:
        update_column_config(columns_config, INDEX_IDENTIFIER, {'hidden': True})
    if data_format in [DataFormat.SET_OF_VALUES, DataFormat.TUPLE_OF_VALUES, DataFormat.LIST_OF_VALUES, DataFormat.NUMPY_LIST, DataFormat.KEY_VALUE_DICT]:
        data_df.rename(columns={0: 'value'}, inplace=True)
    if not isinstance(data_df.index, pd.RangeIndex):
        update_column_config(columns_config, INDEX_IDENTIFIER, {'required': True})