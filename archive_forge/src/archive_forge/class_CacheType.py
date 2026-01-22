from __future__ import annotations
import enum
class CacheType(enum.Enum):
    """The function cache types we implement."""
    DATA = 'DATA'
    RESOURCE = 'RESOURCE'