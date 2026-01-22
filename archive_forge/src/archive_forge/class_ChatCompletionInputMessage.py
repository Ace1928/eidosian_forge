from dataclasses import dataclass
from typing import List, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class ChatCompletionInputMessage(BaseInferenceType):
    content: str
    'The content of the message.'
    role: 'ChatCompletionMessageRole'