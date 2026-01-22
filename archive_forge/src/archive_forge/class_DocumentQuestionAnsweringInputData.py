from dataclasses import dataclass
from typing import Any, List, Optional, Union
from .base import BaseInferenceType
@dataclass
class DocumentQuestionAnsweringInputData(BaseInferenceType):
    """One (document, question) pair to answer"""
    image: Any
    'The image on which the question is asked'
    question: str
    'A question to ask of the document'