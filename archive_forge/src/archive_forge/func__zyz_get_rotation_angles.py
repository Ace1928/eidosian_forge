import numpy as np
import pennylane as qml
from pennylane import math
def _zyz_get_rotation_angles(U):
    """Computes the rotation angles :math:`\\phi`, :math:`\\theta`, :math:`\\omega`
    for a unitary :math:`U` that is :math:`SU(2)`

    Args:
        U (array[complex]): A matrix that is :math:`SU(2)`

    Returns:
        tuple[array[float]]: A tuple containing the rotation angles
            :math:`\\phi`, :math:`\\theta`, :math:`\\omega`

    """
    off_diagonal_elements = math.clip(math.abs(U[:, 0, 1]), 0, 1)
    thetas = 2 * math.arcsin(off_diagonal_elements)
    epsilon = 1e-64
    angles_U00 = math.arctan2(math.imag(U[:, 0, 0]), math.real(U[:, 0, 0]) + epsilon)
    angles_U10 = math.arctan2(math.imag(U[:, 1, 0]), math.real(U[:, 1, 0]) + epsilon)
    phis = -angles_U10 - angles_U00
    omegas = angles_U10 - angles_U00
    phis, thetas, omegas = map(math.squeeze, [phis, thetas, omegas])
    phis = phis % (4 * np.pi)
    thetas = thetas % (4 * np.pi)
    omegas = omegas % (4 * np.pi)
    return (phis, thetas, omegas)