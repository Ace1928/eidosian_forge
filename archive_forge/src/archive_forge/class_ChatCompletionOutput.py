from dataclasses import dataclass
from typing import List, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class ChatCompletionOutput(BaseInferenceType):
    """Outputs for Chat Completion inference"""
    choices: List[ChatCompletionOutputChoice]
    'A list of chat completion choices.'
    created: int
    'The Unix timestamp (in seconds) of when the chat completion was created.'