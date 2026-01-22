import copy
import pickle
import time
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem.Subshape import BuilderUtils, SubshapeObjects
def CombineSubshapes(self, subshape1, subshape2, operation=SubshapeCombineOperations.UNION):
    cs = copy.deepcopy(subshape1)
    if operation == SubshapeCombineOperations.UNION:
        cs.grid |= subshape2.grid
    elif operation == SubshapeCombineOperations.SUM:
        cs.grid += subshape2.grid
    elif operation == SubshapeCombineOperations.INTERSECT:
        cs.grid &= subshape2.grid
    else:
        raise ValueError('bad combination operation')
    return cs