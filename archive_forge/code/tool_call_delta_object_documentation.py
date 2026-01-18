from typing import List, Optional
from typing_extensions import Literal
from ....._models import BaseModel
from .tool_call_delta import ToolCallDelta
An array of tool calls the run step was involved in.

    These can be associated with one of three types of tools: `code_interpreter`,
    `retrieval`, or `function`.
    