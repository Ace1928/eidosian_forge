import networkx as nx
from collections import deque
class CyclicList(list):

    def __getitem__(self, n):
        if isinstance(n, int):
            return list.__getitem__(self, n % len(self))
        elif isinstance(n, slice):
            return list.__getitem__(self, n)

    def succ(self, x):
        return self[(self.index(x) + 1) % len(self)]

    def pred(self, x):
        return self[(self.index(x) - 1) % len(self)]