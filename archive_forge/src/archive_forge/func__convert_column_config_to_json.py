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
def _convert_column_config_to_json(column_config_mapping: ColumnConfigMapping) -> str:
    try:
        return json.dumps({f'{_NUMERICAL_POSITION_PREFIX}{str(k)}' if isinstance(k, int) else k: v for k, v in remove_none_values(column_config_mapping).items()}, allow_nan=False)
    except ValueError as ex:
        raise StreamlitAPIException(f'The provided column config cannot be serialized into JSON: {ex}') from ex