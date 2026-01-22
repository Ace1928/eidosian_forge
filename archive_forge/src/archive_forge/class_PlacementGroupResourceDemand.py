import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
@dataclass
class PlacementGroupResourceDemand(ResourceDemand):
    details: str
    pg_id: Optional[str] = None
    strategy: Optional[str] = None
    state: Optional[str] = None

    def __post_init__(self):
        if not self.details:
            return
        pattern = '^.*:.*\\|.*$'
        match = re.match(pattern, self.details)
        if not match:
            return
        pg_id, details = self.details.split(':')
        strategy, state = details.split('|')
        self.pg_id = pg_id
        self.strategy = strategy
        self.state = state