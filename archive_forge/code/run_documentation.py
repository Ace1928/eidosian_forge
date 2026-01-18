from typing import List, Optional
from typing_extensions import Literal
from ...._models import BaseModel
from .run_status import RunStatus
from ..assistant_tool import AssistantTool
from .required_action_function_tool_call import RequiredActionFunctionToolCall
Usage statistics related to the run.

    This value will be `null` if the run is not in a terminal state (i.e.
    `in_progress`, `queued`, etc.).
    