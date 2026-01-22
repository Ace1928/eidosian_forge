import copy
import pickle
import time
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem.Subshape import BuilderUtils, SubshapeObjects
def GenerateSubshapeShape(self, cmpd, confId=-1, addSkeleton=True, **kwargs):
    shape = SubshapeObjects.ShapeWithSkeleton()
    shape.grid = Geometry.UniformGrid3D(self.gridDims[0], self.gridDims[1], self.gridDims[2], self.gridSpacing)
    AllChem.EncodeShape(cmpd, shape.grid, ignoreHs=False, confId=confId)
    if addSkeleton:
        conf = cmpd.GetConformer(confId)
        self.GenerateSubshapeSkeleton(shape, conf, **kwargs)
    return shape