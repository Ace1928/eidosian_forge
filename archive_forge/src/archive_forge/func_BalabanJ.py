import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def BalabanJ(mol, dMat=None, forceDMat=0):
    """ Calculate Balaban's J value for a molecule

  **Arguments**

    - mol: a molecule

    - dMat: (optional) a distance/adjacency matrix for the molecule, if this
      is not provide, one will be calculated

    - forceDMat: (optional) if this is set, the distance/adjacency matrix
      will be recalculated regardless of whether or not _dMat_ is provided
      or the molecule already has one

  **Returns**

    - a float containing the J value

  We follow the notation of Balaban's paper:
    Chem. Phys. Lett. vol 89, 399-404, (1982)

  """
    if forceDMat or dMat is None:
        if forceDMat:
            dMat = Chem.GetDistanceMatrix(mol, useBO=1, useAtomWts=0, force=1)
            mol._balabanMat = dMat
            adjMat = Chem.GetAdjacencyMatrix(mol, useBO=0, emptyVal=0, force=0, prefix='NoBO')
            mol._adjMat = adjMat
        else:
            try:
                dMat = mol._balabanMat
            except AttributeError:
                dMat = Chem.GetDistanceMatrix(mol, useBO=1, useAtomWts=0, force=0, prefix='Balaban')
                mol._balabanMat = dMat
            try:
                adjMat = mol._adjMat
            except AttributeError:
                adjMat = Chem.GetAdjacencyMatrix(mol, useBO=0, emptyVal=0, force=0, prefix='NoBO')
                mol._adjMat = adjMat
    else:
        adjMat = Chem.GetAdjacencyMatrix(mol, useBO=0, emptyVal=0, force=0, prefix='NoBO')
    s = _VertexDegrees(dMat)
    q = _NumAdjacencies(mol, dMat)
    n = mol.GetNumAtoms()
    mu = q - n + 1
    sum_ = 0.0
    nS = len(s)
    for i in range(nS):
        si = s[i]
        for j in range(i, nS):
            if adjMat[i, j] == 1:
                sum_ += 1.0 / numpy.sqrt(si * s[j])
    if mu + 1 != 0:
        J = float(q) / float(mu + 1) * sum_
    else:
        J = 0
    return J