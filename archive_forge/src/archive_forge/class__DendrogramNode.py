import copy
from abc import abstractmethod
from math import sqrt
from sys import stdout
from nltk.cluster.api import ClusterI
class _DendrogramNode:
    """Tree node of a dendrogram."""

    def __init__(self, value, *children):
        self._value = value
        self._children = children

    def leaves(self, values=True):
        if self._children:
            leaves = []
            for child in self._children:
                leaves.extend(child.leaves(values))
            return leaves
        elif values:
            return [self._value]
        else:
            return [self]

    def groups(self, n):
        queue = [(self._value, self)]
        while len(queue) < n:
            priority, node = queue.pop()
            if not node._children:
                queue.push((priority, node))
                break
            for child in node._children:
                if child._children:
                    queue.append((child._value, child))
                else:
                    queue.append((0, child))
            queue.sort()
        groups = []
        for priority, node in queue:
            groups.append(node.leaves())
        return groups

    def __lt__(self, comparator):
        return cosine_distance(self._value, comparator._value) < 0