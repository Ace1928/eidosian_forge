from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.qp2quad_form.canonicalizers import CANON_METHODS as qp_canon_methods
from cvxpy.reductions.utilities import are_args_affine
Converts a QP to an even more symbolic form.