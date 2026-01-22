import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def _initializeMatrices(self):
    nf = len(self._feats)
    for i in range(1, nf):
        loci = self._feats[i].GetPos()
        for j in range(i):
            locj = self._feats[j].GetPos()
            dist = loci.Distance(locj)
            self._boundsMat[i, j] = dist
            self._boundsMat[j, i] = dist
    for i in range(nf):
        for j in range(i + 1, nf):
            self._boundsMat2D[i, j] = 1000