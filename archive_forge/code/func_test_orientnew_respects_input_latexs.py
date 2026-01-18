from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, zeros)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.vector import (ReferenceFrame, Vector, CoordinateSym,
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import VectorTypeError
from sympy.testing.pytest import raises
import warnings
def test_orientnew_respects_input_latexs():
    N = ReferenceFrame('N')
    q1 = dynamicsymbols('q1')
    A = N.orientnew('a', 'Axis', [q1, N.z])
    def_latex_vecs = ['\\mathbf{\\hat{%s}_%s}' % (A.name.lower(), A.indices[0]), '\\mathbf{\\hat{%s}_%s}' % (A.name.lower(), A.indices[1]), '\\mathbf{\\hat{%s}_%s}' % (A.name.lower(), A.indices[2])]
    name = 'b'
    indices = [x + '1' for x in N.indices]
    new_latex_vecs = ['\\mathbf{\\hat{%s}_{%s}}' % (name.lower(), indices[0]), '\\mathbf{\\hat{%s}_{%s}}' % (name.lower(), indices[1]), '\\mathbf{\\hat{%s}_{%s}}' % (name.lower(), indices[2])]
    B = N.orientnew(name, 'Axis', [q1, N.z], latexs=new_latex_vecs)
    assert A.latex_vecs == def_latex_vecs
    assert B.latex_vecs == new_latex_vecs
    assert B.indices != indices