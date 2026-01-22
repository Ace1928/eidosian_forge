import numpy
from rdkit import RDRandom as random
from rdkit.ML.Data import Quantize
from rdkit.ML.DecTree import ID3, QuantTree
from rdkit.ML.InfoTheory import entropy
def FindBest(resCodes, examples, nBoundsPerVar, nPossibleRes, nPossibleVals, attrs, exIndices=None, **kwargs):
    bestGain = -1000000.0
    best = -1
    bestBounds = []
    if exIndices is None:
        exIndices = list(range(len(examples)))
    if not len(exIndices):
        return (best, bestGain, bestBounds)
    nToTake = kwargs.get('randomDescriptors', 0)
    if nToTake > 0:
        nAttrs = len(attrs)
        if nToTake < nAttrs:
            ids = list(range(nAttrs))
            random.shuffle(ids, random=random.random)
            tmp = [attrs[x] for x in ids[:nToTake]]
            attrs = tmp
    for var in attrs:
        nBounds = nBoundsPerVar[var]
        if nBounds > 0:
            try:
                vTable = [examples[x][var] for x in exIndices]
            except IndexError:
                print('index error retrieving variable: %d' % var)
                raise
            qBounds, gainHere = Quantize.FindVarMultQuantBounds(vTable, nBounds, resCodes, nPossibleRes)
        elif nBounds == 0:
            vTable = ID3.GenVarTable((examples[x] for x in exIndices), nPossibleVals, [var])[0]
            gainHere = entropy.InfoGain(vTable)
            qBounds = []
        else:
            gainHere = -1000000.0
            qBounds = []
        if gainHere > bestGain:
            bestGain = gainHere
            bestBounds = qBounds
            best = var
        elif bestGain == gainHere:
            if len(qBounds) < len(bestBounds):
                best = var
                bestBounds = qBounds
    if best == -1:
        print('best unaltered')
        print('\tattrs:', attrs)
        print('\tnBounds:', numpy.take(nBoundsPerVar, attrs))
        print('\texamples:')
        for example in (examples[x] for x in exIndices):
            print('\t\t', example)
    if 0:
        print('BEST:', len(exIndices), best, bestGain, bestBounds)
        if len(exIndices) < 10:
            print(len(exIndices), len(resCodes), len(examples))
            exs = [examples[x] for x in exIndices]
            vals = [x[best] for x in exs]
            sortIdx = numpy.argsort(vals)
            sortVals = [exs[x] for x in sortIdx]
            sortResults = [resCodes[x] for x in sortIdx]
            for i in range(len(vals)):
                print('   ', i, ['%.4f' % x for x in sortVals[i][1:-1]], sortResults[i])
    return (best, bestGain, bestBounds)