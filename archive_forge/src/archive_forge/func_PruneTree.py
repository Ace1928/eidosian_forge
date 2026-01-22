import copy
import numpy
from rdkit.ML.DecTree import CrossValidate, DecTree
def PruneTree(tree, trainExamples, testExamples, minimizeTestErrorOnly=1):
    """ implements a reduced-error pruning of decision trees

     This algorithm is described on page 69 of Mitchell's book.

     Pruning can be done using just the set of testExamples (the validation set)
     or both the testExamples and the trainExamples by setting minimizeTestErrorOnly
     to 0.

     **Arguments**

       - tree: the initial tree to be pruned

       - trainExamples: the examples used to train the tree

       - testExamples: the examples held out for testing the tree

       - minimizeTestErrorOnly: if this toggle is zero, all examples (i.e.
         _trainExamples_ + _testExamples_ will be used to evaluate the error.

     **Returns**

       a 2-tuple containing:

          1) the best tree

          2) the best error (the one which corresponds to that tree)

    """
    if minimizeTestErrorOnly:
        testSet = testExamples
    else:
        testSet = trainExamples + testExamples
    tree.ClearExamples()
    totErr, badEx = CrossValidate.CrossValidate(tree, testSet, appendExamples=1)
    newTree = _Pruner(tree)
    totErr, badEx = CrossValidate.CrossValidate(newTree, testSet)
    newTree.SetBadExamples(badEx)
    return (newTree, totErr)