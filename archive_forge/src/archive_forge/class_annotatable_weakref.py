from __future__ import annotations
import typing as t
from weakref import ref
from blinker._saferef import BoundMethodWeakref
class annotatable_weakref(ref):
    """A weakref.ref that supports custom instance attributes."""
    receiver_id: t.Optional[IdentityType]
    sender_id: t.Optional[IdentityType]