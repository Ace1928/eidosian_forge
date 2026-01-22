import math
class TanH(ActFunc):
    """ the standard hyperbolic tangent function """

    def Eval(self, x):
        v1 = math.exp(self.beta * x)
        v2 = math.exp(-self.beta * x)
        return (v1 - v2) / (v1 + v2)

    def Deriv(self, x):
        val = self.Eval(x)
        return self.beta * (1 - val * val)

    def DerivFromVal(self, val):
        return self.beta * (1 - val * val)

    def __init__(self, beta=1.0):
        self.beta = beta