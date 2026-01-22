from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
class ClosedSurfaceInCusped(ClosedSurface):

    def __init__(self, manifold, quadvector):
        ClosedSurface.__init__(self, manifold, quadvector)
        self.Incompressible = None
        self.BoundarySlope = None

    def info(self, manifold, out=sys.stdout):
        out.write('ClosedSurfaceInCusped #%d:  Euler %d;  Incompressible %s\n' % (manifold.ClosedSurfaces.index(self), self.EulerCharacteristic, self.Incompressible))
        q, e = self.is_edge_linking_torus()
        if q:
            out.write('    is thin linking surface of edge %s\n' % manifold.Edges[e])
            return
        b, d, t = self.BoundingInfo
        if b == 1:
            out.write('  Bounds %s subcomplex\n' % t)
        elif d == 1:
            out.write('  Double bounds %s subcomplex\n' % t)
        else:
            out.write("  Doesn't bound subcomplex\n")
        for i in range(self.Size):
            quad_weight = self.Coefficients[i]
            if quad_weight > 0:
                weight = '  Quad Type  Q%d3, weight %d' % (self.Quadtypes[i], quad_weight)
            else:
                weight = 'No quads'
            out.write('  In tet %s :  %s\n' % (manifold.Tetrahedra[i], weight))
            out.write('\tTri weights V0: %d V1: %d V2 : %d V3 : %d\n' % (self.get_weight(i, V0), self.get_weight(i, V1), self.get_weight(i, V2), self.get_weight(i, V3)))
            out.write('\n')
        for i in range(len(self.EdgeWeights)):
            out.write('  Edge %s has weight %d\n' % (manifold.Edges[i], self.EdgeWeights[i]))