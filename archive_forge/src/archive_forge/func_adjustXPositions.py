import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
def adjustXPositions(self, pts, data):
    """Return a list of Point() where the x position is set to the nearest x value in *data* for each point in *pts*."""
    points = []
    timeIndices = []
    for p in pts:
        x = np.argwhere(abs(data - p.x()) == abs(data - p.x()).min())
        points.append(Point(data[x], p.y()))
        timeIndices.append(x)
    return (points, timeIndices)