class IntervalTree:
    """
    A data structure that can store (interval, value) pairs and quickly
    retrieve all values for which the interval overlaps with a given
    interval.

    Create an interval tree and add pairs to it::

        sage: from sage.all import RIF
        sage: t = IntervalTree()
        sage: t.insert(RIF(1.01,1.02),'1')
        sage: t.insert(RIF(3.01,3.02),'3')
        sage: t.insert(RIF(2.01,2.02),'2')
        sage: t.insert(RIF(0.99, 3.0),'big')

    Retrieve all values for intervals overlapping [1.5, 2.5]::

        sage: t.find(RIF(1.5, 2.5))
        ['big', '2']

    Calls to insert :py:meth:`insert` and :py:meth:`find` can be mixed::

        sage: t.insert(RIF(4.01,4.02),'4')
        sage: t.find(RIF(1,10))
        ['big', '1', '2', '3', '4']
        sage: t.find(RIF(16, 17))
        []

    This is implemented as interval tree, i.e., as a red-black tree keyed by
    the left endpoint of an interval with each node *N* storing the max of all
    right endpoints of all intervals of the subtree under *N*.

    A demo of red-black trees is at https://www.youtube.com/watch?v=gme8e_6Fnug
    and explanation at http://unhyperbolic.org/rbtree.pdf .
    Also see wikipedia or Cormen et al, *Introduction to Algorithms*.
    """

    class _Node:

        def __init__(self, interval, value):
            self.interval = interval
            self.value = value
            self.max_value = interval.upper()
            self.children = [None, None]
            self.isRed = True

        def update_max_value(self):
            self.max_value = self.interval.upper()
            for child in self.children:
                if child:
                    self.max_value = max(self.max_value, child.max_value)

    def __init__(self):
        self._root = None

    def find(self, interval):
        """
        Finds all values that have been inserted with intervals overlapping the
        given interval which is an element in SageMath's ``RealIntervalField``.

        The runtime of this call is O(log (n) + m) where n is the number of
        intervals in the tree and m the number of the returned intervals.

        The order of the returned values is ascending in the left endpoint of
        the associated intervals. If several intervals have the same left
        endpoint, the order of the returned values depends on the insertion
        order but is still deterministic.
        """
        result = []
        IntervalTree._fill_recursive(self._root, interval, result)
        return result

    @staticmethod
    def _fill_recursive(node, interval, result):
        if not node:
            return
        if node.max_value < interval.lower():
            return
        IntervalTree._fill_recursive(node.children[LEFT], interval, result)
        if node.interval.lower() > interval.upper():
            return
        if node.interval.overlaps(interval):
            result.append(node.value)
        IntervalTree._fill_recursive(node.children[RIGHT], interval, result)

    def insert(self, interval, value):
        """
        Inserts (interval, value) where interval is an element in
        SageMath's ``RealIntervalField`` and value can be anything.
        """
        node = IntervalTree._Node(interval, value)
        if self._root:
            violation, self._root = IntervalTree._insert_fix_and_update_max(self._root, node)
        else:
            self._root = node
        self._root.isRed = False

    @staticmethod
    def _if_red(node):
        if node and node.isRed:
            return node
        return None

    @staticmethod
    def _insert_fix_and_update_max(node, leaf):
        violation, node = IntervalTree._insert_and_fix(node, leaf)
        node.update_max_value()
        return (violation, node)

    @staticmethod
    def _insert_and_fix(node, leaf):
        branch = LEFT if leaf.interval.lower() <= node.interval.lower() else RIGHT
        if not node.children[branch]:
            node.children[branch] = leaf
            return ('VIOLATION' if node.isRed else 'OK', node)
        violation, node.children[branch] = IntervalTree._insert_fix_and_update_max(node.children[branch], leaf)
        if violation == 'OK' or (violation == 'POTENTIAL' and (not node.isRed)):
            return ('OK', node)
        return IntervalTree._fix(node)

    @staticmethod
    def _fix(node):
        if node.isRed:
            return ('VIOLATION', node)
        redChildren = [(branch, child) for branch, child in enumerate(node.children) if IntervalTree._if_red(child)]
        if len(redChildren) == 2:
            for child in node.children:
                child.isRed = False
            node.isRed = True
            return ('POTENTIAL', node)
        branch, child = redChildren[0]
        grandChild = IntervalTree._if_red(child.children[1 - branch])
        if grandChild:
            child.children[1 - branch] = grandChild.children[branch]
            child.update_max_value()
            grandChild.children[branch] = child
            node.children[branch] = grandChild
            child = grandChild
        node.children[branch] = child.children[1 - branch]
        node.update_max_value()
        node.isRed = True
        child.children[1 - branch] = node
        child.isRed = False
        return ('OK', child)