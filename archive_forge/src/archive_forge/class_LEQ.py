from __future__ import annotations
import logging # isort:skip
from typing import (
from ..model import Model
class LEQ(_Operator):
    """ Predicate to test if property values are less than or equal to
    some value.

    Construct and ``LEQ`` predicate as a dict with ``LEQ`` as the key,
    and a value to compare against.

    .. code-block:: python

        # matches any models with .size <= 10
        dict(size={ LEQ: 10 })

    """
    pass