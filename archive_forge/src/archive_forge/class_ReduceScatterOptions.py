from enum import Enum
from dataclasses import dataclass
from datetime import timedelta
@dataclass
class ReduceScatterOptions:
    reduceOp = ReduceOp.SUM
    timeout_ms = unset_timeout_ms