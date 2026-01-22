import pickle
def _exampleCode():
    tree = TreeNode(None, 'root')
    for i in range(3):
        tree.AddChild('child %d' % i)
    print(tree)
    tree.GetChildren()[1].AddChild('grandchild')
    tree.GetChildren()[1].AddChild('grandchild2')
    tree.GetChildren()[1].AddChild('grandchild3')
    print(tree)
    tree.Pickle('save.pkl')
    print('prune')
    tree.PruneChild(tree.GetChildren()[1])
    print('done')
    print(tree)
    import copy
    tree2 = copy.deepcopy(tree)
    print('tree==tree2', tree == tree2)
    foo = [tree]
    print('tree in [tree]:', tree in foo, foo.index(tree))
    print('tree2 in [tree]:', tree2 in foo, foo.index(tree2))
    tree2.GetChildren()[1].AddChild('grandchild4')
    print('tree==tree2', tree == tree2)
    tree.Destroy()