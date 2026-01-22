from types import GenericAlias
class _NodeInfo:
    __slots__ = ('node', 'npredecessors', 'successors')

    def __init__(self, node):
        self.node = node
        self.npredecessors = 0
        self.successors = []