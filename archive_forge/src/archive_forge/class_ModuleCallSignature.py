from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass
class ModuleCallSignature:
    inputs: List[Argument]
    outputs: List[Argument]
    in_spec: str
    out_spec: str