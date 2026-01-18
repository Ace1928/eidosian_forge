from sympy.core.symbol import symbols
from sympy.sets.sets import FiniteSet
from sympy.combinatorics.polyhedron import (Polyhedron,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.testing.pytest import raises
def test_pgroups():
    from sympy.combinatorics.polyhedron import cube, tetrahedron_faces, octahedron_faces, dodecahedron_faces, icosahedron_faces
    from sympy.combinatorics.polyhedron import _pgroup_calcs
    tetrahedron2, cube2, octahedron2, dodecahedron2, icosahedron2, tetrahedron_faces2, cube_faces2, octahedron_faces2, dodecahedron_faces2, icosahedron_faces2 = _pgroup_calcs()
    assert tetrahedron == tetrahedron2
    assert cube == cube2
    assert octahedron == octahedron2
    assert dodecahedron == dodecahedron2
    assert icosahedron == icosahedron2
    assert sorted(map(sorted, tetrahedron_faces)) == sorted(map(sorted, tetrahedron_faces2))
    assert sorted(cube_faces) == sorted(cube_faces2)
    assert sorted(octahedron_faces) == sorted(octahedron_faces2)
    assert sorted(dodecahedron_faces) == sorted(dodecahedron_faces2)
    assert sorted(icosahedron_faces) == sorted(icosahedron_faces2)