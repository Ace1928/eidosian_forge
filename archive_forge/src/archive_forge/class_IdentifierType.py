import sys
class IdentifierType(Node):
    __slots__ = ('names', 'coord', '__weakref__')

    def __init__(self, names, coord=None):
        self.names = names
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    def __iter__(self):
        return
        yield
    attr_names = ('names',)