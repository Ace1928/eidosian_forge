import numpy
from rdkit.ML.DecTree import DecTree
from rdkit.ML.InfoTheory import entropy
def ID3(examples, target, attrs, nPossibleVals, depth=0, maxDepth=-1, **kwargs):
    """ Implements the ID3 algorithm for constructing decision trees.

    From Mitchell's book, page 56

    This is *slightly* modified from Mitchell's book because it supports
      multivalued (non-binary) results.

    **Arguments**

      - examples: a list (nInstances long) of lists of variable values + instance
              values

      - target: an int

      - attrs: a list of ints indicating which variables can be used in the tree

      - nPossibleVals: a list containing the number of possible values of
                   every variable.

      - depth: (optional) the current depth in the tree

      - maxDepth: (optional) the maximum depth to which the tree
                   will be grown

    **Returns**

     a DecTree.DecTreeNode with the decision tree

    **NOTE:** This code cannot bootstrap (start from nothing...)
          use _ID3Boot_ (below) for that.
  """
    varTable = GenVarTable(examples, nPossibleVals, attrs)
    tree = DecTree.DecTreeNode(None, 'node')
    totEntropy = CalcTotalEntropy(examples, nPossibleVals)
    tree.SetData(totEntropy)
    tMat = GenVarTable(examples, nPossibleVals, [target])[0]
    counts = sum(tMat)
    nzCounts = numpy.nonzero(counts)[0]
    if len(nzCounts) == 1:
        res = nzCounts[0]
        tree.SetLabel(res)
        tree.SetName(str(res))
        tree.SetTerminal(1)
    elif len(attrs) == 0 or (maxDepth >= 0 and depth >= maxDepth):
        v = numpy.argmax(counts)
        tree.SetLabel(v)
        tree.SetName('%d?' % v)
        tree.SetTerminal(1)
    else:
        gains = [entropy.InfoGain(x) for x in varTable]
        best = attrs[numpy.argmax(gains)]
        nextAttrs = attrs[:]
        if not kwargs.get('recycleVars', 0):
            nextAttrs.remove(best)
        tree.SetName('Var: %d' % best)
        tree.SetLabel(best)
        tree.SetTerminal(0)
        for val in range(nPossibleVals[best]):
            nextExamples = []
            for example in examples:
                if example[best] == val:
                    nextExamples.append(example)
            if len(nextExamples) == 0:
                v = numpy.argmax(counts)
                tree.AddChild('%d' % v, label=v, data=0.0, isTerminal=1)
            else:
                tree.AddChildNode(ID3(nextExamples, best, nextAttrs, nPossibleVals, depth + 1, maxDepth, **kwargs))
    return tree