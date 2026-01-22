from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Iterator, MutableMapping
from urllib import parse
from streamlit.constants import EMBED_QUERY_PARAMS_KEYS
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
def __set_item_internal(self, key: str, value: str | Iterable[str]) -> None:
    if isinstance(value, dict):
        raise StreamlitAPIException(f'You cannot set a query params key `{key}` to a dictionary.')
    if key in EMBED_QUERY_PARAMS_KEYS:
        raise StreamlitAPIException('Query param embed and embed_options (case-insensitive) cannot be set programmatically.')
    if isinstance(value, Iterable) and (not isinstance(value, str)):
        self._query_params[key] = [str(item) for item in value]
    else:
        self._query_params[key] = str(value)