import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def ClusterTerminalPts(pts, winRad, scale):
    res = []
    tagged = [(y, x) for x, y in enumerate(pts)]
    while tagged:
        head, headIdx = tagged.pop(0)
        currSet = [head]
        i = 0
        while i < len(tagged):
            nbr, nbrIdx = tagged[i]
            if head.location.Distance(nbr.location) < scale * winRad:
                currSet.append(nbr)
                del tagged[i]
            else:
                i += 1
        pt = Geometry.Point3D(0, 0, 0)
        for o in currSet:
            pt += o.location
        pt /= len(currSet)
        res.append(SubshapeObjects.SkeletonPoint(location=pt))
    return res