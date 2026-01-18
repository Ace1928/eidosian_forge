from typing import Union, Optional
from typing_extensions import Literal, Annotated
from ....._utils import PropertyInfo
from ....._models import BaseModel
from .tool_calls_step_details import ToolCallsStepDetails
from .message_creation_step_details import MessageCreationStepDetails
Usage statistics related to the run step.

    This value will be `null` while the run step's status is `in_progress`.
    