from typing import Union, Optional
from typing_extensions import Literal, Annotated
from ....._utils import PropertyInfo
from ....._models import BaseModel
from .tool_calls_step_details import ToolCallsStepDetails
from .message_creation_step_details import MessageCreationStepDetails
class Usage(BaseModel):
    completion_tokens: int
    'Number of completion tokens used over the course of the run step.'
    prompt_tokens: int
    'Number of prompt tokens used over the course of the run step.'
    total_tokens: int
    'Total number of tokens used (prompt + completion).'