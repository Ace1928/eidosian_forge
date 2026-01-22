from __future__ import annotations
import dataclasses
from typing import List, Optional
@dataclasses.dataclass
class PropertyBag(object):
    """Key/value pairs that provide additional information about the object."""
    tags: Optional[List[str]] = dataclasses.field(default=None, metadata={'schema_property_name': 'tags'})