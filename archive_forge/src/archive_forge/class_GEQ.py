from __future__ import annotations
import logging # isort:skip
from typing import (
from ..model import Model
class GEQ(_Operator):
    """ Predicate to test if property values are greater than or equal to
    some value.

    Construct and ``GEQ`` predicate as a dict with ``GEQ`` as the key,
    and a value to compare against.

    .. code-block:: python

        # matches any models with .size >= 10
        dict(size={ GEQ: 10 })

    """
    pass