import copy
import math
import numpy
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw import rdMolDraw2D
def GetSimilarityMapForModel(probeMol, fpFunction, predictionFunction, **kwargs):
    """
    Generates the similarity map for a given ML model and probe molecule,
    and fingerprint function.

    Parameters:
      probeMol -- the probe molecule
      fpFunction -- the fingerprint function
      predictionFunction -- the prediction function of the ML model
      kwargs -- additional arguments for drawing
    """
    weights = GetAtomicWeightsForModel(probeMol, fpFunction, predictionFunction)
    weights, maxWeight = GetStandardizedWeights(weights)
    fig = GetSimilarityMapFromWeights(probeMol, weights, **kwargs)
    return (fig, maxWeight)