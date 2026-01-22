from __future__ import annotations
import warnings
from typing import List, Tuple, TypeVar
import numpy as np
from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
class RelEntrConeQuad(Cone):
    """An approximate construction of the scalar relative entropy cone

    Definition:

    .. math::

        K_{re}=\\text{cl}\\{(x,y,z)\\in\\mathbb{R}_{++}\\times
                \\mathbb{R}_{++}\\times\\mathbb{R}_{++}\\:x\\log(x/y)\\leq z\\}

    Since the above definition is very similar to the ExpCone, we provide a conversion method.

    More details on the approximation can be found in Theorem-3 on page-10 in the paper:
    Semidefinite Approximations of the Matrix Logarithm.

    Parameters
    ----------
    x : Expression
        x in the (approximate) scalar relative entropy cone
    y : Expression
        y in the (approximate) scalar relative entropy cone
    z : Expression
        z in the (approximate) scalar relative entropy cone
    m: Parameter directly related to the number of generated nodes for the quadrature
    approximation used in the algorithm
    k: Another parameter controlling the approximation
    """

    def __init__(self, x: Expression, y: Expression, z: Expression, m: int, k: int, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        self.x = Expression.cast_to_const(x)
        self.y = Expression.cast_to_const(y)
        self.z = Expression.cast_to_const(z)
        args = [self.x, self.y, self.z]
        for val in args:
            if not (val.is_affine() and val.is_real()):
                raise ValueError('All Expression arguments must be affine and real.')
        self.m = m
        self.k = k
        xs, ys, zs = (self.x.shape, self.y.shape, self.z.shape)
        if xs != ys or xs != zs:
            msg = 'All arguments must have the same shapes. Provided arguments haveshapes %s' % str((xs, ys, zs))
            raise ValueError(msg)
        super(RelEntrConeQuad, self).__init__([self.x, self.y, self.z], constr_id)

    def get_data(self):
        return [self.m, self.k, self.id]

    def __str__(self) -> str:
        tup = (self.x, self.y, self.z, self.m, self.k)
        return 'RelEntrConeQuad(%s, %s, %s, %s, %s)' % tup

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def residual(self):
        from cvxpy import Minimize, Problem, Variable, hstack, norm2
        if self.x.value is None or self.y.value is None or self.z.value is None:
            return None
        cvxtypes.expression()
        x = Variable(self.x.shape)
        y = Variable(self.y.shape)
        z = Variable(self.z.shape)
        constr = [RelEntrConeQuad(x, y, z, self.m, self.k)]
        obj = Minimize(norm2(hstack([x, y, z]) - hstack([self.x.value, self.y.value, self.z.value])))
        problem = Problem(obj, constr)
        return problem.solve()

    @property
    def size(self) -> int:
        """The number of entries in the combined cones.
        """
        return 3 * self.num_cones()

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.x.size

    def cone_sizes(self) -> List[int]:
        """The dimensions of the exponential cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        return [3] * self.num_cones()

    def is_dcp(self, dpp: bool=False) -> bool:
        """An exponential constraint is DCP if each argument is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                return all((arg.is_affine() for arg in self.args))
        return all((arg.is_affine() for arg in self.args))

    def is_dgp(self, dpp: bool=False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    @property
    def shape(self) -> Tuple[int, ...]:
        s = (3,) + self.x.shape
        return s

    def save_dual_value(self, value) -> None:
        pass