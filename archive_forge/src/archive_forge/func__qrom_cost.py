import numpy as np
from pennylane.operation import AnyWires, Operation
from pennylane.qchem import factorize
@staticmethod
def _qrom_cost(constants):
    """Return the number of Toffoli gates and the expansion factor needed to implement a QROM
        for the double factorization method.

        The complexity of a QROM computation in the most general form is given by (see Eq. (C39) in
        [`PRX Quantum 2, 030305 (2021) <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`_])

        .. math::

            \\text{cost} = \\left \\lceil \\frac{a + b}{k} \\right \\rceil + \\left \\lceil \\frac{c}{k} \\right
            \\rceil + d \\left ( k + e \\right ),

        where :math:`a, b, c, d, e` are constants that depend on the nature of the QROM
        implementation and the expansion factor :math:`k = 2^n` minimizes the cost. This function
        computes the optimum :math:`k` and the minimum cost for a QROM specification.

        To obtain the optimum values of :math:`k`, we first assume that the cost function is
        continuous and use differentiation to obtain the value of :math:`k` that minimizes the cost.
        This value of :math:`k` is not necessarily an integer power of 2. We then obtain the value
        of :math:`n` as :math:`n = \\log_2(k)` and compute the cost for
        :math:`n_{int}= \\left \\{\\left \\lceil n \\right \\rceil, \\left \\lfloor n \\right \\rfloor \\right \\}`.
        The value of :math:`n_{int}` that gives the smaller cost is used to compute the optimim
        :math:`k`.

        Args:
            constants (tuple[float]): constants specifying a QROM

        Returns:
            tuple(int, int): the cost and the expansion factor for the QROM

        **Example**

        >>> constants = (151.0, 7.0, 151.0, 30.0, -1.0)
        >>> _qrom_cost(constants)
        168, 4
        """
    a, b, c, d, e = constants
    n = np.log2(((a + b + c) / d) ** 0.5)
    k = np.array([2 ** np.floor(n), 2 ** np.ceil(n)])
    cost = np.ceil((a + b) / k) + np.ceil(c / k) + d * (k + e)
    return (int(cost[np.argmin(cost)]), int(k[np.argmin(cost)]))