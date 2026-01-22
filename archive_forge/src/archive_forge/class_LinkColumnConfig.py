from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
class LinkColumnConfig(TypedDict):
    type: Literal['link']
    max_chars: NotRequired[int | None]
    validate: NotRequired[str | None]
    display_text: NotRequired[str | None]