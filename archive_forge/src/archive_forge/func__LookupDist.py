import numpy
from rdkit.ML.Cluster import Clusters
def _LookupDist(dists, i, j, n):
    """ *Internal Use Only*

     returns the distance between points i and j in the symmetric
     distance matrix _dists_

    """
    if i == j:
        return 0.0
    if i > j:
        i, j = (j, i)
    return dists[j * (j - 1) // 2 + i]