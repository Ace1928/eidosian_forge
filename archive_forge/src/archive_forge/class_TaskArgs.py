import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
@generate_doc_dataclass
@dataclass
class TaskArgs:
    """Task-specific parameters."""
    is_regression: Optional[bool] = field(default=None, metadata={'description': 'Text classification specific. Set whether the task is regression (output = one float).'})