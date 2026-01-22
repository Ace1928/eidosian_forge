import numpy
from rdkit.ML.DecTree import DecTree
from rdkit.ML.InfoTheory import entropy
def ID3Boot(examples, attrs, nPossibleVals, initialVar=None, depth=0, maxDepth=-1, **kwargs):
    """ Bootstrapping code for the ID3 algorithm

    see ID3 for descriptions of the arguments

    If _initialVar_ is not set, the algorithm will automatically
     choose the first variable in the tree (the standard greedy
     approach).  Otherwise, _initialVar_ will be used as the first
     split.

  """
    totEntropy = CalcTotalEntropy(examples, nPossibleVals)
    varTable = GenVarTable(examples, nPossibleVals, attrs)
    tree = DecTree.DecTreeNode(None, 'node')
    tree._nResultCodes = nPossibleVals[-1]
    if initialVar is None:
        best = attrs[numpy.argmax([entropy.InfoGain(x) for x in varTable])]
    else:
        best = initialVar
    tree.SetName('Var: %d' % best)
    tree.SetData(totEntropy)
    tree.SetLabel(best)
    tree.SetTerminal(0)
    nextAttrs = list(attrs)
    if not kwargs.get('recycleVars', 0):
        nextAttrs.remove(best)
    for val in range(nPossibleVals[best]):
        nextExamples = []
        for example in examples:
            if example[best] == val:
                nextExamples.append(example)
        tree.AddChildNode(ID3(nextExamples, best, nextAttrs, nPossibleVals, depth, maxDepth, **kwargs))
    return tree