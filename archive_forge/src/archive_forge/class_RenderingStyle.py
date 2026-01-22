import enum
import typing as T
class RenderingStyle(enum.Enum):
    """Rendering style when unparsing parsed docstrings."""
    COMPACT = 1
    CLEAN = 2
    EXPANDED = 3