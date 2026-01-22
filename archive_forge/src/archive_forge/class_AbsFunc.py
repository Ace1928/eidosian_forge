import numpy as np
class AbsFunc(TransformFunction):
    """class for absolute value transformation
    """

    def func(self, x):
        return np.abs(x)

    def inverseplus(self, x):
        return x

    def inverseminus(self, x):
        return 0.0 - x

    def derivplus(self, x):
        return 1.0

    def derivminus(self, x):
        return 0.0 - 1.0