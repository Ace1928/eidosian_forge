from fontTools.varLib.models import VariationModel, supportScalar
from fontTools.designspaceLib import DesignSpaceDocument
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d
from itertools import cycle
import math
import logging
import sys
def _plotLocations2D(model, axis, fig, cols, rows, names, **kwargs):
    subplot = fig.add_subplot(111)
    for i, (support, color, name) in enumerate(zip(model.supports, cycle(pyplot.cm.Set1.colors), cycle(names))):
        if name is not None:
            subplot.set_title(name)
        subplot.set_xlabel(axis)
        pyplot.xlim(-1.0, +1.0)
        Xs = support.get(axis, (-1.0, 0.0, +1.0))
        X, Y = ([], [])
        for x in stops(Xs):
            y = supportScalar({axis: x}, support)
            X.append(x)
            Y.append(y)
        subplot.plot(X, Y, color=color, **kwargs)
        _plotLocationsDots(model.locations, [axis], subplot)