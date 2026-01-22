from rdkit import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers as rdsimdiv
def ClusterBits(self, corrMat):
    distMat = 1 / corrMat
    pkr = rdsimdiv.HierarchicalClusterPicker(self._type)
    cls = pkr.Cluster(distMat, len(self._bidList), self._nClusters)
    self._clusters = []
    for cl in cls:
        self._clusters.append([self._bidList[i] for i in cl])