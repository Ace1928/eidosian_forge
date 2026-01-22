import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def FindGridPointBetweenPoints(pt1, pt2, shapeGrid, winRad):
    center = pt1 + pt2
    center /= 2.0
    d = 100000000.0
    while d > shapeGrid.GetSpacing():
        count, centroid = Geometry.ComputeGridCentroid(shapeGrid, center, winRad)
        d = center.Distance(centroid)
        center = centroid
    return center