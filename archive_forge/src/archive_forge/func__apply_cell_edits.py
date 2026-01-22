from __future__ import annotations
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import (
from typing_extensions import TypeAlias
from streamlit import logger as _logger
from streamlit import type_util
from streamlit.deprecation_util import deprecate_func_name
from streamlit.elements.form import current_form_id
from streamlit.elements.lib.column_config_utils import (
from streamlit.elements.lib.pandas_styler_utils import marshall_styler
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import DataFormat, DataFrameGenericAlias, Key, is_type, to_key
from streamlit.util import calc_md5
def _apply_cell_edits(df: pd.DataFrame, edited_rows: Mapping[int, Mapping[str, str | int | float | bool | None]], dataframe_schema: DataframeSchema) -> None:
    """Apply cell edits to the provided dataframe (inplace).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to apply the cell edits to.

    edited_rows : Mapping[int, Mapping[str, str | int | float | bool | None]]
        A hierarchical mapping based on row position -> column name -> value

    dataframe_schema: DataframeSchema
        The schema of the dataframe.
    """
    for row_id, row_changes in edited_rows.items():
        row_pos = int(row_id)
        for col_name, value in row_changes.items():
            if col_name == INDEX_IDENTIFIER:
                df.index.values[row_pos] = _parse_value(value, dataframe_schema[INDEX_IDENTIFIER])
            else:
                col_pos = df.columns.get_loc(col_name)
                df.iat[row_pos, col_pos] = _parse_value(value, dataframe_schema[col_name])