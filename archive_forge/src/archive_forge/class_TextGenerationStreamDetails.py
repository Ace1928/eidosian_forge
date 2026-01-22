from dataclasses import dataclass
from typing import List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TextGenerationStreamDetails(BaseInferenceType):
    """Generation details. Only available when the generation is finished."""
    finish_reason: 'TextGenerationFinishReason'
    'The reason why the generation was stopped.'
    generated_tokens: int
    'The number of generated tokens'
    seed: int
    'The random seed used for generation'