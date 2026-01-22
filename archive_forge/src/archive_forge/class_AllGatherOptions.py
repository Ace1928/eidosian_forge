from enum import Enum
from dataclasses import dataclass
from datetime import timedelta
@dataclass
class AllGatherOptions:
    timeout_ms = unset_timeout_ms