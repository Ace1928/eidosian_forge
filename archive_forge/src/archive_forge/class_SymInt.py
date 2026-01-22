from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass(repr=False)
class SymInt(_Union):
    as_expr: SymExpr
    as_int: int