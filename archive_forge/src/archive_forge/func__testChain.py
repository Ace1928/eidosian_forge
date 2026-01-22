import copy
import numpy
from rdkit.ML.DecTree import CrossValidate, DecTree
def _testChain():
    from rdkit.ML.DecTree import ID3
    oPts = [[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 0, 1]]
    tPts = oPts
    tree = ID3.ID3Boot(oPts, attrs=range(len(oPts[0]) - 1), nPossibleVals=[2] * len(oPts[0]))
    tree.Print()
    err, _ = CrossValidate.CrossValidate(tree, oPts)
    print('original error:', err)
    err, _ = CrossValidate.CrossValidate(tree, tPts)
    print('original holdout error:', err)
    newTree, frac2 = PruneTree(tree, oPts, tPts)
    newTree.Print()
    print('best error of pruned tree:', frac2)
    err, badEx = CrossValidate.CrossValidate(newTree, tPts)
    print('pruned holdout error is:', err)
    print(badEx)