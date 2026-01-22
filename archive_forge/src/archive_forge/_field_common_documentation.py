from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect

    Create a checked ``PMap`` field.

    :param key: The required type for the keys of the map.
    :param value: The required type for the values of the map.
    :param optional: If true, ``None`` can be used as a value for
        this field.
    :param invariant: Pass-through to ``field``.

    :return: A ``field`` containing a ``CheckedPMap``.
    