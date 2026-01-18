import operator
import weakref
import numpy as np
from .. import functions as fn
from .. import colormap
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapButton import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients
def tickValue(self, tick):
    """Return the value (from 0.0 to 1.0) of the specified tick.
        
        ==============  ==================================================================
        **Arguments:**
        tick            Can be either an integer corresponding to the index of the tick
                        or a Tick object. Ex: if you had a slider with 3 ticks and you
                        wanted the value of the middle tick, the index would be 1.
        ==============  ==================================================================
        """
    tick = self.getTick(tick)
    return self.ticks[tick]