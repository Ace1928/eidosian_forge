from __future__ import annotations
from contextlib import nullcontext
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any, Collection, Literal, Sequence, cast
import streamlit.elements.arrow_vega_lite as arrow_vega_lite
from streamlit import type_util
from streamlit.color_util import (
from streamlit.elements.altair_utils import AddRowsMetadata
from streamlit.elements.arrow import Data
from streamlit.elements.utils import last_index_for_melted_dataframes
from streamlit.errors import Error, StreamlitAPIException
from streamlit.proto.ArrowVegaLiteChart_pb2 import (
from streamlit.runtime.metrics_util import gather_metrics
def _parse_x_column(df: pd.DataFrame, x_from_user: str | None) -> str | None:
    if x_from_user is None:
        return None
    elif isinstance(x_from_user, str):
        if x_from_user not in df.columns:
            raise StreamlitColumnNotFoundError(df, x_from_user)
        return x_from_user
    else:
        raise StreamlitAPIException(f"x parameter should be a column name (str) or None to use the  dataframe's index. Value given: {x_from_user} (type {type(x_from_user)})")