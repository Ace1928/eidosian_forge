from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass
class GraphArgument:
    name: str
    graph: 'Graph'