from __future__ import annotations
import logging # isort:skip
from typing import (
from ..model import Model
class OR(_Operator):
    """ Form disjunctions from other query predicates.

    Construct an ``OR`` expression by making a dict with ``OR`` as the key,
    and a list of other query expressions as the value:

    .. code-block:: python

        # matches any Axis subclasses or models with .name == "mycircle"
        { OR: [dict(type=Axis), dict(name="mycircle")] }

    """
    pass