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
def ConstrainedEnum(matches, mol, pcophore, bounds, use2DLimits=False, index=0, soFar=[]):
    """ Enumerates the list of atom mappings a molecule
  has to a particular pharmacophore.
  We do check distance bounds here.


  """
    nMatches = len(matches)
    if index >= nMatches:
        yield (soFar, [])
    elif index == nMatches - 1:
        for entry in matches[index]:
            nextStep = soFar + [entry]
            if index != 0:
                atomMatch = _checkMatch(nextStep, mol, bounds, pcophore, use2DLimits)
            else:
                atomMatch = ChemicalFeatures.GetAtomMatch(nextStep)
            if atomMatch:
                yield (soFar + [entry], atomMatch)
    else:
        for entry in matches[index]:
            nextStep = soFar + [entry]
            if index != 0:
                atomMatch = _checkMatch(nextStep, mol, bounds, pcophore, use2DLimits)
                if not atomMatch:
                    continue
            for val in ConstrainedEnum(matches, mol, pcophore, bounds, use2DLimits=use2DLimits, index=index + 1, soFar=nextStep):
                if val:
                    yield val