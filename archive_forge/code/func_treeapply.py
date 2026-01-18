from functools import partial
from sympy.strategies import chain, minimize
from sympy.strategies.core import identity
import sympy.strategies.branch as branch
from sympy.strategies.branch import yieldify
def treeapply(tree, join, leaf=identity):
    """ Apply functions onto recursive containers (tree).

    Explanation
    ===========

    join - a dictionary mapping container types to functions
      e.g. ``{list: minimize, tuple: chain}``

    Keys are containers/iterables.  Values are functions [a] -> a.

    Examples
    ========

    >>> from sympy.strategies.tree import treeapply
    >>> tree = [(3, 2), (4, 1)]
    >>> treeapply(tree, {list: max, tuple: min})
    2

    >>> add = lambda *args: sum(args)
    >>> def mul(*args):
    ...     total = 1
    ...     for arg in args:
    ...         total *= arg
    ...     return total
    >>> treeapply(tree, {list: mul, tuple: add})
    25
    """
    for typ in join:
        if isinstance(tree, typ):
            return join[typ](*map(partial(treeapply, join=join, leaf=leaf), tree))
    return leaf(tree)