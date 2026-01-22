from __future__ import annotations
import typing as t
from collections import defaultdict
from contextlib import contextmanager
from inspect import iscoroutinefunction
from warnings import warn
from weakref import WeakValueDictionary
from blinker._utilities import annotatable_weakref
from blinker._utilities import hashable_identity
from blinker._utilities import IdentityType
from blinker._utilities import lazy_property
from blinker._utilities import reference
from blinker._utilities import symbol
from blinker._utilities import WeakTypes
def _extract_sender(self, sender: t.Any) -> t.Any:
    if not self.receivers:
        if __debug__ and sender and (len(sender) > 1):
            raise TypeError(f'send() accepts only one positional argument, {len(sender)} given')
        return []
    if len(sender) == 0:
        sender = None
    elif len(sender) > 1:
        raise TypeError(f'send() accepts only one positional argument, {len(sender)} given')
    else:
        sender = sender[0]
    return sender