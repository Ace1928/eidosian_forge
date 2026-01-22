from .basic import Basic
from .sorting import ordered
from .sympify import sympify
from sympy.utilities.iterables import iterable
class preorder_traversal:
    """
    Do a pre-order traversal of a tree.

    This iterator recursively yields nodes that it has visited in a pre-order
    fashion. That is, it yields the current node then descends through the
    tree breadth-first to yield all of a node's children's pre-order
    traversal.


    For an expression, the order of the traversal depends on the order of
    .args, which in many cases can be arbitrary.

    Parameters
    ==========
    node : SymPy expression
        The expression to traverse.
    keys : (default None) sort key(s)
        The key(s) used to sort args of Basic objects. When None, args of Basic
        objects are processed in arbitrary order. If key is defined, it will
        be passed along to ordered() as the only key(s) to use to sort the
        arguments; if ``key`` is simply True then the default keys of ordered
        will be used.

    Yields
    ======
    subtree : SymPy expression
        All of the subtrees in the tree.

    Examples
    ========

    >>> from sympy import preorder_traversal, symbols
    >>> x, y, z = symbols('x y z')

    The nodes are returned in the order that they are encountered unless key
    is given; simply passing key=True will guarantee that the traversal is
    unique.

    >>> list(preorder_traversal((x + y)*z, keys=None)) # doctest: +SKIP
    [z*(x + y), z, x + y, y, x]
    >>> list(preorder_traversal((x + y)*z, keys=True))
    [z*(x + y), z, x + y, x, y]

    """

    def __init__(self, node, keys=None):
        self._skip_flag = False
        self._pt = self._preorder_traversal(node, keys)

    def _preorder_traversal(self, node, keys):
        yield node
        if self._skip_flag:
            self._skip_flag = False
            return
        if isinstance(node, Basic):
            if not keys and hasattr(node, '_argset'):
                args = node._argset
            else:
                args = node.args
            if keys:
                if keys != True:
                    args = ordered(args, keys, default=False)
                else:
                    args = ordered(args)
            for arg in args:
                yield from self._preorder_traversal(arg, keys)
        elif iterable(node):
            for item in node:
                yield from self._preorder_traversal(item, keys)

    def skip(self):
        """
        Skip yielding current node's (last yielded node's) subtrees.

        Examples
        ========

        >>> from sympy import preorder_traversal, symbols
        >>> x, y, z = symbols('x y z')
        >>> pt = preorder_traversal((x + y*z)*z)
        >>> for i in pt:
        ...     print(i)
        ...     if i == x + y*z:
        ...             pt.skip()
        z*(x + y*z)
        z
        x + y*z
        """
        self._skip_flag = True

    def __next__(self):
        return next(self._pt)

    def __iter__(self):
        return self