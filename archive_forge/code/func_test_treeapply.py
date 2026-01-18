from sympy.strategies.tree import treeapply, greedy, allresults, brute
from functools import partial, reduce
def test_treeapply():
    tree = ([3, 3], [4, 1], 2)
    assert treeapply(tree, {list: min, tuple: max}) == 3
    assert treeapply(tree, {list: add, tuple: mul}) == 60