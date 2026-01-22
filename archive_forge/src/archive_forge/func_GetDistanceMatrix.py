import pickle
import numpy
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols, MolSimilarity
from rdkit.ML.Cluster import Murtagh
def GetDistanceMatrix(data, metric, isSimilarity=1):
    """ data should be a list of tuples with fingerprints in position 1
   (the rest of the elements of the tuple are not important)

    Returns the symmetric distance matrix
    (see ML.Cluster.Resemblance for layout documentation)

  """
    nPts = len(data)
    distsMatrix = numpy.zeros(nPts * (nPts - 1) // 2, dtype=numpy.float64)
    nSoFar = 0
    for col in range(1, nPts):
        fp1 = data[col][1]
        nBits1 = fp1.GetNumBits()
        for row in range(col):
            fp2 = data[row][1]
            nBits2 = fp2.GetNumBits()
            if nBits1 > nBits2:
                fp1 = DataStructs.FoldFingerprint(fp1, nBits1 / nBits2)
            elif nBits2 > nBits1:
                fp2 = DataStructs.FoldFingerprint(fp2, nBits2 / nBits1)
            if isSimilarity:
                distsMatrix[nSoFar] = 1.0 - metric(fp1, fp2)
            else:
                distsMatrix[nSoFar] = metric(fp1, fp2)
            nSoFar += 1
    return distsMatrix