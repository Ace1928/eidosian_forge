from dataclasses import dataclass
from typing import List, Optional
@dataclass
class GridPos:
    x: int
    y: int
    w: int
    h: int