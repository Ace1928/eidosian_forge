import weakref
from math import atan2, degrees
from ..functions import clip_scalar
from ..Qt import QtCore, QtWidgets
from . import ArrowItem
from .GraphicsObject import GraphicsObject
Position can be set either as an index referring to the sample number or
        the position 0.0 - 1.0
        If *rotate* is True, then the item rotates to match the tangent of the curve.
        