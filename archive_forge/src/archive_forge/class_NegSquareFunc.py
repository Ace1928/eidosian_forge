import numpy as np
class NegSquareFunc(TransformFunction):
    """negative quadratic function

    """

    def func(self, x):
        return -np.power(x, 2)

    def inverseplus(self, x):
        return np.sqrt(-x)

    def inverseminus(self, x):
        return 0.0 - np.sqrt(-x)

    def derivplus(self, x):
        return 0.0 - 0.5 / np.sqrt(-x)

    def derivminus(self, x):
        return 0.5 / np.sqrt(-x)