from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
from .chat_completion_token_logprob import ChatCompletionTokenLogprob
class ChoiceDeltaToolCall(BaseModel):
    index: int
    id: Optional[str] = None
    'The ID of the tool call.'
    function: Optional[ChoiceDeltaToolCallFunction] = None
    type: Optional[Literal['function']] = None
    'The type of the tool. Currently, only `function` is supported.'