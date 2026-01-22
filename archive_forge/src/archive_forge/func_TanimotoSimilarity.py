import numpy
from rdkit.Chem.rdReducedGraphs import *
def TanimotoSimilarity(arr1, arr2):
    numer = arr1.dot(arr2)
    if numer == 0.0:
        return 0.0
    denom = arr1.dot(arr1) + arr2.dot(arr2) - numer
    if denom == 0.0:
        return 0.0
    return numer / denom