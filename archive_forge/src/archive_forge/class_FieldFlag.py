from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class FieldFlag(IntFlag):
    """TABLE 8.70 Field flags common to all field types."""
    READ_ONLY = 1
    REQUIRED = 2
    NO_EXPORT = 4