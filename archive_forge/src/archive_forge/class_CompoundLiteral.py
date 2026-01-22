import sys
class CompoundLiteral(Node):
    __slots__ = ('type', 'init', 'coord', '__weakref__')

    def __init__(self, type, init, coord=None):
        self.type = type
        self.init = init
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(('type', self.type))
        if self.init is not None:
            nodelist.append(('init', self.init))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
        if self.init is not None:
            yield self.init
    attr_names = ()