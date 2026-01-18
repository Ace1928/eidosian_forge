from __future__ import annotations
from typing import Any, Dict
Merge two nested dictionaries.

    Effectively a recursive ``dict.update``.

    Examples
    --------
    Merge two flat dictionaries:
    >>> nested_update(
    ...     {'a': 1, 'b': 2},
    ...     {'b': 3, 'c': 4}
    ... )
    {'a': 1, 'b': 3, 'c': 4}

    Merge two nested dictionaries:
    >>> nested_update(
    ...     {'x': {'a': 1, 'b': 2}, 'y': 5, 'z': 6},
    ...     {'x': {'b': 3, 'c': 4}, 'z': 7, '0': 8},
    ... )
    {'x': {'a': 1, 'b': 3, 'c': 4}, 'y': 5, 'z': 7, '0': 8}

    