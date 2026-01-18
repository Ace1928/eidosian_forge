from .arrow import eArrow
from .simplex import *
from .tetrahedron import Tetrahedron
import re
def write_SnapPea_file(mcomplex, fileobject):
    out = fileobject.write
    if hasattr(fileobject, 'name'):
        name = fileobject.name
    else:
        name = 'untitled'
    out('% Triangulation\n\n' + name + '\nnot_attempted 0.0\nunknown_orientability\nCS_unknown\n\n')
    torus_cusps = []
    for vertex in mcomplex.Vertices:
        g = vertex.link_genus()
        if g > 1:
            raise ValueError('Link of vertex has genus more than 1.')
        if g == 1:
            torus_cusps.append(vertex)
    out('%d 0\n' % len(torus_cusps))
    for i in torus_cusps:
        out('   torus   0.000000000000   0.000000000000\n')
    out('\n')
    out('%d\n' % len(mcomplex))
    for tet in mcomplex.Tetrahedra:
        for face in TwoSubsimplices:
            out('    %d' % mcomplex.Tetrahedra.index(tet.Neighbor[face]))
        out('\n')
        for face in TwoSubsimplices:
            out(' %d%d%d%d' % tet.Gluing[face].tuple())
        out('\n')
        for vert in ZeroSubsimplices:
            vertex = tet.Class[vert]
            if vertex.link_genus() == 1:
                out('%d ' % torus_cusps.index(vertex))
            else:
                out('-1 ')
        out('\n')
        if hasattr(tet, 'PeripheralCurves'):
            for curve in tet.PeripheralCurves:
                for sheet in curve:
                    for v in ZeroSubsimplices:
                        for f in TwoSubsimplices:
                            out('%d ' % sheet[v][f])
                        if v == V3:
                            out('\n')
                        else:
                            out('  ')
        else:
            for i in range(4):
                out('0 0 0 0  0 0 0 0   0 0 0 0   0 0 0 0\n')
        out('0.0 0.0\n\n')