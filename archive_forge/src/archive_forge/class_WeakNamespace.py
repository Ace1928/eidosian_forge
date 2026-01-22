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
class WeakNamespace(WeakValueDictionary):
    """A weak mapping of signal names to signals.

    Automatically cleans up unused Signals when the last reference goes out
    of scope.  This namespace implementation exists for a measure of legacy
    compatibility with Blinker <= 1.2, and may be dropped in the future.

    .. versionadded:: 1.3

    """

    def signal(self, name: str, doc: str | None=None) -> NamedSignal:
        """Return the :class:`NamedSignal` *name*, creating it if required.

        Repeated calls to this function will return the same signal object.

        """
        try:
            return self[name]
        except KeyError:
            result = self.setdefault(name, NamedSignal(name, doc))
            return result