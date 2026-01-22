import math
import sys
import time
import numpy
import rdkit.DistanceGeometry as DG
from rdkit import Chem
from rdkit import RDLogger as logging
from rdkit.Chem import ChemicalFeatures, ChemicalForceFields
from rdkit.Chem import rdDistGeom as MolDG
from rdkit.Chem.Pharm3D import ExcludedVolume
from rdkit.ML.Data import Stats
def _checkMatch(match, mol, bounds, pcophore, use2DLimits):
    """ **INTERNAL USE ONLY**

  checks whether a particular atom match can be satisfied by
  a molecule

  """
    atomMatch = ChemicalFeatures.GetAtomMatch(match)
    if not atomMatch:
        return None
    elif use2DLimits:
        if not Check2DBounds(atomMatch, mol, pcophore):
            return None
    if not CoarseScreenPharmacophore(atomMatch, bounds, pcophore):
        return None
    return atomMatch