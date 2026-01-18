from __future__ import annotations
import logging # isort:skip
import colorsys
from abc import ABCMeta, abstractmethod
from math import sqrt
from re import match
from typing import TYPE_CHECKING, Union
from ..core.serialization import AnyRep, Serializable, Serializer
from ..util.deprecation import deprecated
def to_hsl(self) -> HSL:
    """ Return a HSL copy for this HSL color.

        Returns:
            :class:`~bokeh.colors.HSL`

        """
    return self.copy()