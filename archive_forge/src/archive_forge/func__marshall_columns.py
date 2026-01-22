from __future__ import annotations
from typing import TYPE_CHECKING, Any
from streamlit import type_util
from streamlit.elements.lib import pandas_styler_utils
from streamlit.proto.Components_pb2 import ArrowTable as ArrowTableProto
def _marshall_columns(proto: ArrowTableProto, columns: Series) -> None:
    """Marshall pandas.DataFrame columns into an ArrowTable proto.

    Parameters
    ----------
    proto : proto.ArrowTable
        Output. The protobuf for a Streamlit ArrowTable proto.

    columns : Series
        Column labels to use for resulting frame.
        Will default to RangeIndex (0, 1, 2, ..., n) if no column labels are provided.

    """
    import pandas as pd
    columns = map(type_util.maybe_tuple_to_list, columns.values)
    columns_df = pd.DataFrame(columns)
    proto.columns = type_util.data_frame_to_bytes(columns_df)