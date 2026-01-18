from __future__ import annotations
from typing import List, Union
from typing_extensions import Literal, Required, TypedDict
from ..._types import FileTypes
The timestamp granularities to populate for this transcription.

    `response_format` must be set `verbose_json` to use timestamp granularities.
    Either or both of these options are supported: `word`, or `segment`. Note: There
    is no additional latency for segment timestamps, but generating word timestamps
    incurs additional latency.
    