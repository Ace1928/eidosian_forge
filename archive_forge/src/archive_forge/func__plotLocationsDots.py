from fontTools.varLib.models import VariationModel, supportScalar
from fontTools.designspaceLib import DesignSpaceDocument
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d
from itertools import cycle
import math
import logging
import sys
def _plotLocationsDots(locations, axes, subplot, **kwargs):
    for loc, color in zip(locations, cycle(pyplot.cm.Set1.colors)):
        if len(axes) == 1:
            subplot.plot([loc.get(axes[0], 0)], [1.0], 'o', color=color, **kwargs)
        elif len(axes) == 2:
            subplot.plot([loc.get(axes[0], 0)], [loc.get(axes[1], 0)], [1.0], 'o', color=color, **kwargs)
        else:
            raise AssertionError(len(axes))