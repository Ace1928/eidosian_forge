from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RDF, pi, matrix, block_matrix, vector
def _adaptive_newton_step(hyperbolicStructure, errors_with_norm, verbose=False):
    errors, errors_norm = errors_with_norm
    num_edges = len(hyperbolicStructure.mcomplex.Edges)
    penalties, penalty_derivative = _large_angle_penalties_and_derivatives(hyperbolicStructure, verbose=verbose)
    all_errors = vector(errors + penalties)
    jacobian = hyperbolicStructure.jacobian()
    penalty_derivative_matrix = matrix(RDF, penalty_derivative, ncols=num_edges)
    m = block_matrix([[jacobian], [penalty_derivative_matrix]])
    mInv = _pseudo_inverse(m, verbose=verbose)
    mInvErrs = mInv * all_errors
    for i in range(14):
        step_size = RDF(0.5) ** i
        new_edge_params = list(vector(hyperbolicStructure.edge_lengths) - step_size * mInvErrs)
        try:
            newHyperbolicStructure = HyperbolicStructure(hyperbolicStructure.mcomplex, new_edge_params)
        except BadDihedralAngleError:
            continue
        new_errors_with_norm = _compute_errors_with_norm(newHyperbolicStructure)
        if new_errors_with_norm[1] < errors_norm:
            return (newHyperbolicStructure, new_errors_with_norm)
    raise NewtonStepError()