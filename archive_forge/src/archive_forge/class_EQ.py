from __future__ import annotations
import logging # isort:skip
from typing import (
from ..model import Model
class EQ(_Operator):
    """ Predicate to test if property values are equal to some value.

    Construct and ``EQ`` predicate as a dict with ``EQ`` as the key,
    and a value to compare against.

    .. code-block:: python

        # matches any models with .size == 10
        dict(size={ EQ: 10 })

    """
    pass