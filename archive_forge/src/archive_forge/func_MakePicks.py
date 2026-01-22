import bisect
from rdkit import DataStructs
from rdkit.DataStructs.TopNContainer import TopNContainer
def MakePicks(self, force=False, silent=False):
    if self._picks is not None and (not force):
        return
    nProbes = len(self.probes)
    scores = [None] * nProbes
    for i in range(nProbes):
        scores[i] = []
    j = 0
    fps = []
    for origFp in self.data:
        for i in range(nProbes):
            score = DataStructs.FingerprintSimilarity(self.probes[i], origFp, self.simMetric)
            bisect.insort(scores[i], (score, j))
            if len(scores[i]) >= self.numToPick:
                del scores[self.numToPick:]
        if self.onlyNames and hasattr(origFp, '_fieldsFromDb'):
            fps.append(origFp._fieldsFromDb[0])
        else:
            fps.append(origFp)
        j += 1
        if not silent and (not j % 1000):
            print('scored %d fps' % j)
    nPicked = 0
    self._picks = []
    taken = [0] * len(fps)
    while nPicked < self.numToPick:
        rowIdx = nPicked % len(scores)
        row = scores[rowIdx]
        score, idx = row.pop()
        while taken[idx] and len(row):
            score, idx = row.pop()
        if not taken[idx]:
            fp = fps[idx]
            self._picks.append((fp, score))
            taken[idx] = 1
            nPicked += 1