from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RDF, pi, matrix, block_matrix, vector
def _large_angle_penalties_and_derivatives(hyperbolicStructure, verbose=False):
    global _theta_thres
    global _theta_max
    penalties = []
    penalty_derivatives = []
    number_large_angles = 0
    max_angle = 0
    for tet, m in enumerate(hyperbolicStructure.dihedral_angles):
        for j in range(1, 4):
            for i in range(0, j):
                if m[i][j] > _theta_thres:
                    penalties.append(m[i][j] - _theta_max)
                    penalty_derivatives.append(hyperbolicStructure.derivative_of_single_dihedral_angle(tet, i, j))
                    number_large_angles += 1
                    if m[i][j] > max_angle:
                        max_angle = m[i][j]
    if verbose:
        print('Number of large angles: %d, Maximum: %f' % (number_large_angles, max_angle))
    return (penalties, penalty_derivatives)