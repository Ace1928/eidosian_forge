import collections
from typing import Any, Optional
def _prune_dict(self, max_size: int) -> None:
    if len(self) >= max_size:
        diff = len(self) - max_size
        for k in list(self.keys())[:diff]:
            del self[k]