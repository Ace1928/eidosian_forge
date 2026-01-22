from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict
def _accountForHybridisation(self, atom):
    """Calculates the hybridisation score for a single atom in a molecule"""
    return self._hybridisations[atom.GetHybridization()]