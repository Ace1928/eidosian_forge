from rdkit import DataStructs
from rdkit.DataStructs import TopNContainer
class TopNScreener(SimilarityScreener):
    """ A screener that only returns the top N hits found

      **Notes**

        - supports forward iteration and getitem

    """

    def __init__(self, num, **kwargs):
        SimilarityScreener.__init__(self, **kwargs)
        self.numToGet = num
        self.topN = None
        self._pos = 0

    def Reset(self):
        self._pos = 0

    def __iter__(self):
        if self.topN is None:
            self._initTopN()
        self.Reset()
        return self

    def next(self):
        if self._pos >= self.numToGet:
            raise StopIteration
        else:
            res = self.topN[self._pos]
            self._pos += 1
            return res
    __next__ = next

    def _initTopN(self):
        self.topN = TopNContainer.TopNContainer(self.numToGet)
        for obj in self.dataSource:
            fp = self.fingerprinter(obj)
            sim = DataStructs.FingerprintSimilarity(fp, self.probe, self.metric)
            self.topN.Insert(sim, obj)

    def __len__(self):
        if self.topN is None:
            self._initTopN()
        return self.numToGet

    def __getitem__(self, idx):
        if self.topN is None:
            self._initTopN()
        return self.topN[idx]