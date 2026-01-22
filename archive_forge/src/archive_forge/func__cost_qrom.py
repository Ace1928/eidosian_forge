import numpy as np
from scipy import integrate
from pennylane.operation import AnyWires, Operation
@staticmethod
def _cost_qrom(lz):
    """Return the minimum number of Toffoli gates needed for erasing the output of a QROM.

        Args:
            lz (int): sum of the atomic numbers

        Returns:
            int: the minimum cost of erasing the output of a QROM

        **Example**

        >>> lz = 100
        >>> _cost_qrom(lz)
        21
        """
    if lz <= 0 or not isinstance(lz, (int, np.integer)):
        raise ValueError('The sum of the atomic numbers must be a positive integer.')
    k_f = np.floor(np.log2(lz) / 2)
    k_c = np.ceil(np.log2(lz) / 2)
    cost_f = int(2 ** k_f + np.ceil(2 ** (-1 * k_f) * lz))
    cost_c = int(2 ** k_c + np.ceil(2 ** (-1 * k_c) * lz))
    return min(cost_f, cost_c)