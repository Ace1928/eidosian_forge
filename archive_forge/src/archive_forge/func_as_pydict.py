from collections.abc import Mapping
from typing import Any
from ray.util.annotations import Deprecated
def as_pydict(self) -> dict:
    """
        Convert to a normal Python dict. This will create a new copy of the row."""
    return dict(self.items())