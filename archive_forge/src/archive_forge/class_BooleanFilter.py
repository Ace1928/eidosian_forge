from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class BooleanFilter(Filter):
    """ A ``BooleanFilter`` filters data by returning the subset of data corresponding to indices
    where the values of the booleans array is True.
    """
    booleans = Nullable(Seq(Bool), help='\n    A list of booleans indicating which rows of data to select.\n    ')

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and 'booleans' not in kwargs:
            kwargs['booleans'] = args[0]
        super().__init__(**kwargs)