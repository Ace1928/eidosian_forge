import sys
class Typename(Node):
    __slots__ = ('name', 'quals', 'align', 'type', 'coord', '__weakref__')

    def __init__(self, name, quals, align, type, coord=None):
        self.name = name
        self.quals = quals
        self.align = align
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(('type', self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
    attr_names = ('name', 'quals', 'align')