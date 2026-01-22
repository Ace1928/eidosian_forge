from __future__ import annotations
from typing import Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
class RunSubmitToolOutputsParamsNonStreaming(RunSubmitToolOutputsParamsBase):
    stream: Optional[Literal[False]]
    '\n    If `true`, returns a stream of events that happen during the Run as server-sent\n    events, terminating when the Run enters a terminal state with a `data: [DONE]`\n    message.\n    '