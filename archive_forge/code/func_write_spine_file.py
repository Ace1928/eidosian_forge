from .arrow import eArrow
from .simplex import *
from .tetrahedron import Tetrahedron
import re
def write_spine_file(mcomplex, fileobject):
    out = fileobject.write
    for edge in mcomplex.Edges:
        n = edge.valence()
        A = edge.get_arrow()
        tets, global_faces, local_faces, back_local_faces = ([], [], [], [])
        for i in range(n):
            tets.append(A.Tetrahedron.Index + 1)
            global_faces.append(A.Tetrahedron.Class[A.Face].Index + 1)
            local_faces.append(A.Face)
            back_local_faces.append(comp(A.head()))
            A.next()
        signs = [1 if (tets[i], local_faces[i]) < (tets[(i + 1) % n], back_local_faces[(i + 1) % n]) else -1 for i in range(n)]
        ans = repr([signs[i] * global_faces[i] for i in range(n)])[1:-1].replace(',', '')
        out(ans + '\n')