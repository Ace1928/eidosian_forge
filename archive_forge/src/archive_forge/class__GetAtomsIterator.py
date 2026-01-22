from rdkit import DataStructs, RDConfig, rdBase
from rdkit.Chem import rdchem
from rdkit.Geometry import rdGeometry
from rdkit.Chem.inchi import *
from rdkit.Chem.rdchem import *
from rdkit.Chem.rdCIPLabeler import *
from rdkit.Chem.rdmolfiles import *
from rdkit.Chem.rdmolops import *
class _GetAtomsIterator(_GetRDKitObjIterator):

    def _sizeCalc(self):
        return self._mol.GetNumAtoms()

    def _getRDKitItem(self, i):
        return self._mol.GetAtomWithIdx(i)