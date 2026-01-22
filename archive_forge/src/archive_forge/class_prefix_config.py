from dataclasses import dataclass, field
from typing import List
@dataclass
class prefix_config:
    num_virtual_tokens: int = 30
    task_type: str = 'CAUSAL_LM'