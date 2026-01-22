import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def Grow(self, examples, attrs, nPossibleVals, nTries=10, pruneIt=0, lessGreedy=0):
    """ Grows the forest by adding trees

     **Arguments**

      - examples: the examples to be used for training

      - attrs: a list of the attributes to be used in training

      - nPossibleVals: a list with the number of possible values each variable
        (as well as the result) can take on

      - nTries: the number of new trees to add

      - pruneIt: a toggle for whether or not the tree should be pruned

      - lessGreedy: toggles the use of a less greedy construction algorithm where
        each possible tree root is used.  The best tree from each step is actually
        added to the forest.

    """
    self._nPossible = nPossibleVals
    for i in range(nTries):
        tree, frac = CrossValidate.CrossValidationDriver(examples, attrs, nPossibleVals, silent=1, calcTotalError=1, lessGreedy=lessGreedy)
        if pruneIt:
            tree, frac2 = PruneTree.PruneTree(tree, tree.GetTrainingExamples(), tree.GetTestExamples(), minimizeTestErrorOnly=0)
            print('prune: ', frac, frac2)
            frac = frac2
        self.AddTree(tree, frac)
        if i % (nTries / 10) == 0:
            print('Cycle: % 4d' % i)