import sys
import warnings
from collections import namedtuple
import numpy
from rdkit import DataStructs, ForceField, RDConfig, rdBase
from rdkit.Chem import *
from rdkit.Chem.ChemicalFeatures import *
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
from rdkit.Chem.rdChemReactions import *
from rdkit.Chem.rdDepictor import *
from rdkit.Chem.rdDistGeom import *
from rdkit.Chem.rdFingerprintGenerator import *
from rdkit.Chem.rdForceFieldHelpers import *
from rdkit.Chem.rdMolAlign import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.rdMolEnumerator import *
from rdkit.Chem.rdMolTransforms import *
from rdkit.Chem.rdPartialCharges import *
from rdkit.Chem.rdqueries import *
from rdkit.Chem.rdReducedGraphs import *
from rdkit.Chem.rdShapeHelpers import *
from rdkit.Geometry import rdGeometry
from rdkit.RDLogger import logger
def GetConformerRMSMatrix(mol, atomIds=None, prealigned=False):
    """ Returns the RMS matrix of the conformers of a molecule.
    As a side-effect, the conformers will be aligned to the first
    conformer (i.e. the reference) and will left in the aligned state.

    Arguments:
      - mol:     the molecule
      - atomIds: (optional) list of atom ids to use a points for
                 alingment - defaults to all atoms
      - prealigned: (optional) by default the conformers are assumed
                    be unaligned and will therefore be aligned to the
                    first conformer

    Note that the returned RMS matrix is symmetrical, i.e. it is the
    lower half of the matrix, e.g. for 5 conformers::

      rmsmatrix = [ a,
                    b, c,
                    d, e, f,
                    g, h, i, j]

    where a is the RMS between conformers 0 and 1, b is the RMS between
    conformers 0 and 2, etc.
    This way it can be directly used as distance matrix in e.g. Butina
    clustering.

    """
    rmsvals = []
    confIds = [conf.GetId() for conf in mol.GetConformers()]
    if not prealigned:
        if atomIds:
            AlignMolConformers(mol, atomIds=atomIds, RMSlist=rmsvals)
        else:
            AlignMolConformers(mol, RMSlist=rmsvals)
    else:
        for i in range(1, len(confIds)):
            rmsvals.append(GetConformerRMS(mol, confIds[0], confIds[i], atomIds=atomIds, prealigned=prealigned))
    cmat = []
    for i in range(1, len(confIds)):
        cmat.append(rmsvals[i - 1])
        for j in range(1, i):
            cmat.append(GetConformerRMS(mol, confIds[i], confIds[j], atomIds=atomIds, prealigned=True))
    return cmat