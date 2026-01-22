import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
@dataclass
class TrainingExample:
    prompt: str
    response: str
    violated_category_codes: List[str]
    label: Literal['safe', 'unsafe']
    explanation: Optional[str] = None