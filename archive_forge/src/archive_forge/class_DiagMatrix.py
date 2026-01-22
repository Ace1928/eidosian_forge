from sympy.core.sympify import _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.core import S, Eq, Ge
from sympy.core.mul import Mul
from sympy.functions.special.tensor_functions import KroneckerDelta
class DiagMatrix(MatrixExpr):
    """
    Turn a vector into a diagonal matrix.
    """

    def __new__(cls, vector):
        vector = _sympify(vector)
        obj = MatrixExpr.__new__(cls, vector)
        shape = vector.shape
        dim = shape[1] if shape[0] == 1 else shape[0]
        if vector.shape[0] != 1:
            obj._iscolumn = True
        else:
            obj._iscolumn = False
        obj._shape = (dim, dim)
        obj._vector = vector
        return obj

    @property
    def shape(self):
        return self._shape

    def _entry(self, i, j, **kwargs):
        if self._iscolumn:
            result = self._vector._entry(i, 0, **kwargs)
        else:
            result = self._vector._entry(0, j, **kwargs)
        if i != j:
            result *= KroneckerDelta(i, j)
        return result

    def _eval_transpose(self):
        return self

    def as_explicit(self):
        from sympy.matrices.dense import diag
        return diag(*list(self._vector.as_explicit()))

    def doit(self, **hints):
        from sympy.assumptions import ask, Q
        from sympy.matrices.expressions.matmul import MatMul
        from sympy.matrices.expressions.transpose import Transpose
        from sympy.matrices.dense import eye
        from sympy.matrices.matrices import MatrixBase
        vector = self._vector
        if ask(Q.diagonal(vector)):
            return vector
        if isinstance(vector, MatrixBase):
            ret = eye(max(vector.shape))
            for i in range(ret.shape[0]):
                ret[i, i] = vector[i]
            return type(vector)(ret)
        if vector.is_MatMul:
            matrices = [arg for arg in vector.args if arg.is_Matrix]
            scalars = [arg for arg in vector.args if arg not in matrices]
            if scalars:
                return Mul.fromiter(scalars) * DiagMatrix(MatMul.fromiter(matrices).doit()).doit()
        if isinstance(vector, Transpose):
            vector = vector.arg
        return DiagMatrix(vector)