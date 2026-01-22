import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _CreateBondDictEtc(mol, numAtoms):
    """ _Internal Use Only_
     Used by BertzCT

  """
    bondDict = {}
    nList = [None] * numAtoms
    vdList = [0] * numAtoms
    for aBond in mol.GetBonds():
        atom1 = aBond.GetBeginAtomIdx()
        atom2 = aBond.GetEndAtomIdx()
        if atom1 > atom2:
            atom2, atom1 = (atom1, atom2)
        if not aBond.GetIsAromatic():
            bondDict[atom1, atom2] = aBond.GetBondType()
        else:
            bondDict[atom1, atom2] = Chem.BondType.AROMATIC
        if nList[atom1] is None:
            nList[atom1] = [atom2]
        elif atom2 not in nList[atom1]:
            nList[atom1].append(atom2)
        if nList[atom2] is None:
            nList[atom2] = [atom1]
        elif atom1 not in nList[atom2]:
            nList[atom2].append(atom1)
    for i, element in enumerate(nList):
        try:
            element.sort()
            vdList[i] = len(element)
        except Exception:
            vdList[i] = 0
    return (bondDict, nList, vdList)