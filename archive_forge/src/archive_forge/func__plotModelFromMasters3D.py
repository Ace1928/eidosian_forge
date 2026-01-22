from fontTools.varLib.models import VariationModel, supportScalar
from fontTools.designspaceLib import DesignSpaceDocument
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d
from itertools import cycle
import math
import logging
import sys
def _plotModelFromMasters3D(model, masterValues, fig, **kwargs):
    assert len(model.axisOrder) == 2
    axis1, axis2 = (model.axisOrder[0], model.axisOrder[1])
    axis1_min = min((loc.get(axis1, 0) for loc in model.locations))
    axis1_max = max((loc.get(axis1, 0) for loc in model.locations))
    axis2_min = min((loc.get(axis2, 0) for loc in model.locations))
    axis2_max = max((loc.get(axis2, 0) for loc in model.locations))
    import numpy as np
    X = np.arange(axis1_min, axis1_max, (axis1_max - axis1_min) / 100)
    Y = np.arange(axis2_min, axis2_max, (axis2_max - axis2_min) / 100)
    X, Y = np.meshgrid(X, Y)
    Z = []
    for row_x, row_y in zip(X, Y):
        z_row = []
        Z.append(z_row)
        for x, y in zip(row_x, row_y):
            loc = {axis1: x, axis2: y}
            v = model.interpolateFromMasters(loc, masterValues)
            z_row.append(v)
    Z = np.array(Z)
    axis3D = fig.add_subplot(111, projection='3d')
    axis3D.plot_surface(X, Y, Z, **kwargs)