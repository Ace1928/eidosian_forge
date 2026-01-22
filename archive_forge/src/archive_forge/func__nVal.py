import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _nVal(atom):
    return ptable.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetTotalNumHs()