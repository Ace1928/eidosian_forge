from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
class TextColumnConfig(TypedDict):
    type: Literal['text']
    max_chars: NotRequired[int | None]
    validate: NotRequired[str | None]