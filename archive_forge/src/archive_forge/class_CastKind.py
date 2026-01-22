from __future__ import annotations
import enum
from .types import Type, Bool, Uint
class CastKind(enum.Enum):
    """A return value indicating the type of cast that can occur from one type to another."""
    EQUAL = enum.auto()
    'The two types are equal; no cast node is required at all.'
    IMPLICIT = enum.auto()
    "The 'from' type can be cast to the 'to' type implicitly.  A :class:`~.expr.Cast` node with\n    ``implicit==True`` is the minimum required to specify this."
    LOSSLESS = enum.auto()
    "The 'from' type can be cast to the 'to' type explicitly, and the cast will be lossless.  This\n    requires a :class:`~.expr.Cast`` node with ``implicit=False``, but there's no danger from\n    inserting one."
    DANGEROUS = enum.auto()
    "The 'from' type has a defined cast to the 'to' type, but depending on the value, it may lose\n    data.  A user would need to manually specify casts."
    NONE = enum.auto()
    "There is no casting permitted from the 'from' type to the 'to' type."