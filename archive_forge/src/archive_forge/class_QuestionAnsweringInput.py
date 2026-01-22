from dataclasses import dataclass
from typing import Optional
from .base import BaseInferenceType
@dataclass
class QuestionAnsweringInput(BaseInferenceType):
    """Inputs for Question Answering inference"""
    inputs: QuestionAnsweringInputData
    'One (context, question) pair to answer'
    parameters: Optional[QuestionAnsweringParameters] = None
    'Additional inference parameters'