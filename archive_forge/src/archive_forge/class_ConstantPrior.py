import numpy as np
class ConstantPrior(Prior):
    """Constant prior, with energy = constant and zero forces

    Parameters:

    constant: energy value for the constant.

    Example:

    >>> from ase.optimize import GPMin
    >>> from ase.optimize.gpmin.prior import ConstantPrior
    >>> op = GPMin(atoms, Prior = ConstantPrior(10)
    """

    def __init__(self, constant):
        self.constant = constant
        Prior.__init__(self)

    def potential(self, x):
        d = x.shape[0]
        output = np.zeros(d + 1)
        output[0] = self.constant
        return output

    def set_constant(self, constant):
        self.constant = constant