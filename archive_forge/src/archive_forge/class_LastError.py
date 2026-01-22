from typing import Union, Optional
from typing_extensions import Literal, Annotated
from ....._utils import PropertyInfo
from ....._models import BaseModel
from .tool_calls_step_details import ToolCallsStepDetails
from .message_creation_step_details import MessageCreationStepDetails
class LastError(BaseModel):
    code: Literal['server_error', 'rate_limit_exceeded']
    'One of `server_error` or `rate_limit_exceeded`.'
    message: str
    'A human-readable description of the error.'