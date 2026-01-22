import copy
import math
import numpy
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw import rdMolDraw2D
def GetAtomicWeightsForFingerprint(refMol, probeMol, fpFunction, metric=DataStructs.DiceSimilarity):
    """
    Calculates the atomic weights for the probe molecule
    based on a fingerprint function and a metric.

    Parameters:
      refMol -- the reference molecule
      probeMol -- the probe molecule
      fpFunction -- the fingerprint function
      metric -- the similarity metric

    Note:
      If fpFunction needs additional parameters, use a lambda construct
    """
    _DeleteFpInfoAttr(probeMol)
    _DeleteFpInfoAttr(refMol)
    refFP = fpFunction(refMol, -1)
    baseSimilarity = metric(refFP, fpFunction(probeMol, -1))
    weights = [baseSimilarity - metric(refFP, fpFunction(probeMol, atomId)) for atomId in range(probeMol.GetNumAtoms())]
    _DeleteFpInfoAttr(probeMol)
    _DeleteFpInfoAttr(refMol)
    return weights