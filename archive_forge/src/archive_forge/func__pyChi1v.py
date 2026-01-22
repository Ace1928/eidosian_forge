import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyChi1v(mol):
    """  From equations (5),(11) and (12) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  """
    deltas = numpy.array(_hkDeltas(mol, skipHs=0))
    res = 0.0
    for bond in mol.GetBonds():
        v = deltas[bond.GetBeginAtomIdx()] * deltas[bond.GetEndAtomIdx()]
        if v != 0.0:
            res += numpy.sqrt(1.0 / v)
    return res