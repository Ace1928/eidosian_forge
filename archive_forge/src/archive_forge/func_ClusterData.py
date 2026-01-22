import numpy
from rdkit.ML.Cluster import Clusters
def ClusterData(data, nPts, method, isDistData=0):
    """  clusters the data points passed in and returns the cluster tree

      **Arguments**

        - data: a list of lists (or array, or whatever) with the input
          data (see discussion of _isDistData_ argument for the exception)

        - nPts: the number of points to be used

        - method: determines which clustering algorithm should be used.
            The defined constants for these are:
            'WARDS, SLINK, CLINK, UPGMA'

        - isDistData: set this toggle when the data passed in is a
            distance matrix.  The distance matrix should be stored
            symmetrically so that _LookupDist (above) can retrieve
            the results:
              for i<j: d_ij = dists[j*(j-1)//2 + i]


      **Returns**

        - a single entry list with the cluster tree
    """
    data = numpy.array(data)
    if not isDistData:
        sz = data.shape[1]
        ia, ib, crit = MurtaghCluster(data, nPts, sz, method)
    else:
        ia, ib, crit = MurtaghDistCluster(data, nPts, method)
    c = _ToClusters(data, nPts, ia, ib, crit, isDistData=isDistData)
    return c