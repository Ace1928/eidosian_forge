from typing import List, Optional
from typing_extensions import Literal
from ...._models import BaseModel
from .message_content import MessageContent
class IncompleteDetails(BaseModel):
    reason: Literal['content_filter', 'max_tokens', 'run_cancelled', 'run_expired', 'run_failed']
    'The reason the message is incomplete.'