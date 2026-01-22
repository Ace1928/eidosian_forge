import math
import numpy
from rdkit import Chem, Geometry
def _findHydAtoms(nbrs, atomNames):
    return [nid for nid in nbrs if atomNames[nid] == 'H']