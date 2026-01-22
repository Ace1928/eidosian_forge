from typing import List, Tuple
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import (
from cvxpy.constraints.zero import Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers.von_neumann_entr_canon import (
class QuadApprox(Canonicalization):
    CANON_METHODS = {RelEntrConeQuad: RelEntrConeQuad_canon, OpRelEntrConeQuad: OpRelEntrConeQuad_canon}

    def __init__(self, problem=None) -> None:
        super(QuadApprox, self).__init__(problem=problem, canon_methods=QuadApprox.CANON_METHODS)