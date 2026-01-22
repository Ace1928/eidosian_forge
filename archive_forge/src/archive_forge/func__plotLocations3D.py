from fontTools.varLib.models import VariationModel, supportScalar
from fontTools.designspaceLib import DesignSpaceDocument
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d
from itertools import cycle
import math
import logging
import sys
def _plotLocations3D(model, axes, fig, rows, cols, names, **kwargs):
    ax1, ax2 = axes
    axis3D = fig.add_subplot(111, projection='3d')
    for i, (support, color, name) in enumerate(zip(model.supports, cycle(pyplot.cm.Set1.colors), cycle(names))):
        if name is not None:
            axis3D.set_title(name)
        axis3D.set_xlabel(ax1)
        axis3D.set_ylabel(ax2)
        pyplot.xlim(-1.0, +1.0)
        pyplot.ylim(-1.0, +1.0)
        Xs = support.get(ax1, (-1.0, 0.0, +1.0))
        Ys = support.get(ax2, (-1.0, 0.0, +1.0))
        for x in stops(Xs):
            X, Y, Z = ([], [], [])
            for y in Ys:
                z = supportScalar({ax1: x, ax2: y}, support)
                X.append(x)
                Y.append(y)
                Z.append(z)
            axis3D.plot(X, Y, Z, color=color, **kwargs)
        for y in stops(Ys):
            X, Y, Z = ([], [], [])
            for x in Xs:
                z = supportScalar({ax1: x, ax2: y}, support)
                X.append(x)
                Y.append(y)
                Z.append(z)
            axis3D.plot(X, Y, Z, color=color, **kwargs)
        _plotLocationsDots(model.locations, [ax1, ax2], axis3D)