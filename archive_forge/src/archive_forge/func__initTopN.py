from rdkit import DataStructs
from rdkit.DataStructs import TopNContainer
def _initTopN(self):
    self.topN = TopNContainer.TopNContainer(self.numToGet)
    for obj in self.dataSource:
        fp = self.fingerprinter(obj)
        sim = DataStructs.FingerprintSimilarity(fp, self.probe, self.metric)
        self.topN.Insert(sim, obj)