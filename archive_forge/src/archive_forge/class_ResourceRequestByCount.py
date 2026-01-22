import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
@dataclass
class ResourceRequestByCount:
    bundle: Dict[str, float]
    count: int

    def __str__(self) -> str:
        return f'[{self.count} {self.bundle}]'