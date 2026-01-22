from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum
class LlamaGuardVersion(Enum):
    LLAMA_GUARD_1 = 'Llama Guard 1'
    LLAMA_GUARD_2 = 'Llama Guard 2'