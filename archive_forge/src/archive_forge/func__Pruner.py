import copy
import numpy
from rdkit.ML.DecTree import CrossValidate, DecTree
def _Pruner(node, level=0):
    """Recursively finds and removes the nodes whose removals improve classification

       **Arguments**

         - node: the tree to be pruned.  The pruning data should already be contained
           within node (i.e. node.GetExamples() should return the pruning data)

         - level: (optional) the level of recursion, used only in _verbose printing


       **Returns**

          the pruned version of node


       **Notes**

        - This uses a greedy algorithm which basically does a DFS traversal of the tree,
          removing nodes whenever possible.

        - If removing a node does not affect the accuracy, it *will be* removed.  We
          favor smaller trees.

    """
    if _verbose:
        print('  ' * level, '<%d>  ' % level, '>>> Pruner')
    children = node.GetChildren()[:]
    bestTree = copy.deepcopy(node)
    bestErr = 1000000.0
    for i in range(len(children)):
        child = children[i]
        examples = child.GetExamples()
        if _verbose:
            print('  ' * level, '<%d>  ' % level, ' Child:', i, child.GetLabel())
            bestTree.Print()
            print()
        if len(examples):
            if _verbose:
                print('  ' * level, '<%d>  ' % level, '  Examples', len(examples))
            if child.GetTerminal():
                if _verbose:
                    print('  ' * level, '<%d>  ' % level, '    Terminal')
                continue
            if _verbose:
                print('  ' * level, '<%d>  ' % level, '    Nonterminal')
            workTree = copy.deepcopy(bestTree)
            newNode = _Pruner(child, level=level + 1)
            workTree.ReplaceChildIndex(i, newNode)
            tempErr = _GetLocalError(workTree)
            if tempErr <= bestErr:
                bestErr = tempErr
                bestTree = copy.deepcopy(workTree)
                if _verbose:
                    print('  ' * level, '<%d>  ' % level, '>->->->->->')
                    print('  ' * level, '<%d>  ' % level, 'replacing:', i, child.GetLabel())
                    child.Print()
                    print('  ' * level, '<%d>  ' % level, 'with:')
                    newNode.Print()
                    print('  ' * level, '<%d>  ' % level, '<-<-<-<-<-<')
            else:
                workTree.ReplaceChildIndex(i, child)
            bestGuess = MaxCount(child.GetExamples())
            newNode = DecTree.DecTreeNode(workTree, 'L:%d' % bestGuess, label=bestGuess, isTerminal=1)
            newNode.SetExamples(child.GetExamples())
            workTree.ReplaceChildIndex(i, newNode)
            if _verbose:
                print('  ' * level, '<%d>  ' % level, 'ATTEMPT:')
                workTree.Print()
            newErr = _GetLocalError(workTree)
            if _verbose:
                print('  ' * level, '<%d>  ' % level, '---> ', newErr, bestErr)
            if newErr <= bestErr:
                bestErr = newErr
                bestTree = copy.deepcopy(workTree)
                if _verbose:
                    print('  ' * level, '<%d>  ' % level, 'PRUNING:')
                    workTree.Print()
            else:
                if _verbose:
                    print('  ' * level, '<%d>  ' % level, 'FAIL')
                workTree.ReplaceChildIndex(i, child)
        else:
            if _verbose:
                print('  ' * level, '<%d>  ' % level, '  No Examples', len(examples))
            pass
    if _verbose:
        print('  ' * level, '<%d>  ' % level, '<<< out')
    return bestTree