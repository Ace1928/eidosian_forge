import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def unlinkFromView(self):
    """Unlink this axis from a ViewBox."""
    oldView = self.linkedView()
    self._linkedView = None
    if self.orientation in ['right', 'left']:
        if oldView is not None:
            oldView.sigYRangeChanged.disconnect(self.linkedViewChanged)
    elif oldView is not None:
        oldView.sigXRangeChanged.disconnect(self.linkedViewChanged)
    if oldView is not None:
        oldView.sigResized.disconnect(self.linkedViewChanged)