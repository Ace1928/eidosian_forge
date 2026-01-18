from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Instance, Required
from .transforms import Transform
 Represent a composition of two scales, which useful for defining
    sub-coordinate systems.

    