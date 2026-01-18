from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypeVar
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
Create a new pandas.DataFrame where display values are used instead of original ones.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe with original values.

    styles : dict
        pandas.Styler translated styles.

    