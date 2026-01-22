from rdkit import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers as rdsimdiv
def MapToClusterFP(self, fp):
    """ Map the fingerprint to a smaller sized (= number of clusters) fingerprint

        Each cluster get a bit in the new fingerprint and is turned on if any of the bits in
        the cluster are turned on in the original fingerprint"""
    ebv = DataStructs.ExplicitBitVect(self._nClusters)
    for i, cls in enumerate(self._clusters):
        for bid in cls:
            if fp[bid]:
                ebv.SetBit(i)
                break
    return ebv