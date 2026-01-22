import math
from rdkit import RDLogger as logging
from rdkit import Geometry
from rdkit.Chem.Features import FeatDirUtilsRD as FeatDirUtils
import os
import sys
from optparse import OptionParser
from rdkit import RDConfig
def _getVectNormal(v, tol=0.0001):
    if math.fabs(v.x) > tol:
        res = Geometry.Point3D(v.y, -v.x, 0)
    elif math.fabs(v.y) > tol:
        res = Geometry.Point3D(-v.y, v.x, 0)
    elif math.fabs(v.z) > tol:
        res = Geometry.Point3D(1, 0, 0)
    else:
        raise ValueError('cannot find normal to the null vector')
    res.Normalize()
    return res