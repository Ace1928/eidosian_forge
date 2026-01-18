from typing import List, Optional, Union
import numpy as np
from cvxpy.atoms import (
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.constraints.exponential import OpRelEntrConeQuad
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
def von_neumann_entr_canon(expr: von_neumann_entr, real_args: List[Union[Expression, None]], imag_args: List[Union[Expression, None]], real2imag):
    """
    The von Neumann entropy of X is sum(entr(eigvals(X)).
    Each eigenvalue of X appears twice as an eigenvalue of the Hermitian dilation of X.
    """
    canon_expr = expand_and_reapply(expr, real_args[0], imag_args[0])
    if imag_args[0] is not None:
        canon_expr /= 2
    return (canon_expr, None)