from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .base import BaseInferenceType
@dataclass
class TableQuestionAnsweringInputData(BaseInferenceType):
    """One (table, question) pair to answer"""
    question: str
    'The question to be answered about the table'
    table: Dict[str, List[str]]
    'The table to serve as context for the questions'