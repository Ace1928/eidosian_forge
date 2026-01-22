from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.qp2quad_form.canonicalizers import CANON_METHODS as qp_canon_methods
from cvxpy.reductions.utilities import are_args_affine
class Qp2SymbolicQp(Canonicalization):
    """
    Reduces a quadratic problem to a problem that consists of affine
    expressions and symbolic quadratic forms.
    """

    def __init__(self, problem=None) -> None:
        super(Qp2SymbolicQp, self).__init__(problem=problem, canon_methods=qp_canon_methods)

    def accepts(self, problem):
        """
        Problems with quadratic, piecewise affine objectives,
        piecewise-linear constraints inequality constraints, and
        affine equality constraints are accepted.
        """
        return accepts(problem)

    def apply(self, problem):
        """Converts a QP to an even more symbolic form."""
        if not self.accepts(problem):
            raise ValueError('Cannot reduce problem to symbolic QP')
        return super(Qp2SymbolicQp, self).apply(problem)