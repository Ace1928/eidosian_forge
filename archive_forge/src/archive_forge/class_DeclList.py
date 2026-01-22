import sys
class DeclList(Node):
    __slots__ = ('decls', 'coord', '__weakref__')

    def __init__(self, decls, coord=None):
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(('decls[%d]' % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in self.decls or []:
            yield child
    attr_names = ()