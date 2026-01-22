import copy
import pickle
import time
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem.Subshape import BuilderUtils, SubshapeObjects
def SampleSubshape(self, subshape1, newSpacing):
    ogrid = subshape1.grid
    rgrid = Geometry.UniformGrid3D(self.gridDims[0], self.gridDims[1], self.gridDims[2], newSpacing)
    for idx in range(rgrid.GetSize()):
        l = rgrid.GetGridPointLoc(idx)
        v = ogrid.GetValPoint(l)
        rgrid.SetVal(idx, v)
    res = SubshapeObjects.ShapeWithSkeleton()
    res.grid = rgrid
    return res