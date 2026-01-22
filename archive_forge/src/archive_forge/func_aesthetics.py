from __future__ import annotations
import typing
from copy import deepcopy
import pandas as pd
from .._utils import (
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError
from ..layer import layer
from ..mapping import aes
from abc import ABC
@classmethod
def aesthetics(cls) -> set[str]:
    """
        Return a set of all non-computed aesthetics for this stat.

        stats should not override this method.
        """
    aesthetics = cls.REQUIRED_AES.copy()
    calculated = aes(**cls.DEFAULT_AES)._calculated
    for ae in set(cls.DEFAULT_AES) - set(calculated):
        aesthetics.add(ae)
    return aesthetics