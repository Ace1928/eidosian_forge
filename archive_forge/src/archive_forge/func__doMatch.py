import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def _doMatch(inv, atoms):
    """ Helper function to check if all atoms in the list are the same
      
      Arguments:
      - inv:    atom invariants (used to define equivalence of atoms)
      - atoms:  list of atoms to check

      Return: boolean
  """
    match = True
    for i in range(len(atoms) - 1):
        for j in range(i + 1, len(atoms)):
            if inv[atoms[i].GetIdx()] != inv[atoms[j].GetIdx()]:
                match = False
                return match
    return match