import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def visualizeChiralSubstituentsGrid(mol):
    idxChiral = Chem.FindMolChiralCenters(mol)[0][0]
    return visualizeSubstituentsGrid(mol, idxChiral)