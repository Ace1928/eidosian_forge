from dataclasses import dataclass
from typing import Optional
from .base import BaseInferenceType
@dataclass
class QuestionAnsweringOutputElement(BaseInferenceType):
    """Outputs of inference for the Question Answering task"""
    answer: str
    'The answer to the question.'
    end: int
    'The character position in the input where the answer ends.'
    score: float
    'The probability associated to the answer.'
    start: int
    'The character position in the input where the answer begins.'