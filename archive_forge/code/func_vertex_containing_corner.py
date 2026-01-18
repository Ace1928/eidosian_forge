from collections import OrderedDict
from ... import sage_helper
def vertex_containing_corner(self, corner):
    return self._vertex_containing_corner[corner.triangle, corner.vertex]