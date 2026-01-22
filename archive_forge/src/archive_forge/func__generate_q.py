from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
from sympy.physics.mechanics.method import _Methods
from sympy.core.backend import Matrix
def _generate_q(self):
    q_ind = []
    for joint in self._joints:
        for coordinate in joint.coordinates:
            if coordinate in q_ind:
                raise ValueError('Coordinates of joints should be unique.')
            q_ind.append(coordinate)
    return Matrix(q_ind)