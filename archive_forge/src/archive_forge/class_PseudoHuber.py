import numpy as np
class PseudoHuber(Penalty):
    """
    The pseudo-Huber penalty.
    """

    def __init__(self, dlt, weights=1.0):
        super().__init__(weights)
        self.dlt = dlt

    def func(self, params):
        v = np.sqrt(1 + (params / self.dlt) ** 2)
        v -= 1
        v *= self.dlt ** 2
        return np.sum(self.weights * self.alpha * v, 0)

    def deriv(self, params):
        v = np.sqrt(1 + (params / self.dlt) ** 2)
        return params * self.weights * self.alpha / v

    def deriv2(self, params):
        v = np.power(1 + (params / self.dlt) ** 2, -3 / 2)
        return self.weights * self.alpha * v