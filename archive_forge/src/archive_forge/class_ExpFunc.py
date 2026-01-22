import numpy as np
class ExpFunc(TransformFunction):

    def func(self, x):
        return np.exp(x)

    def inverse(self, y):
        return np.log(y)

    def deriv(self, x):
        return np.exp(x)