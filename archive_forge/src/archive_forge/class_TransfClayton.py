import warnings
import numpy as np
from scipy.special import expm1, gamma
class TransfClayton(Transforms):

    def _checkargs(self, theta):
        return theta > 0

    def evaluate(self, t, theta):
        return np.power(t, -theta) - 1.0

    def inverse(self, phi, theta):
        return np.power(1 + phi, -1 / theta)

    def deriv(self, t, theta):
        return -theta * np.power(t, -theta - 1)

    def deriv2(self, t, theta):
        return theta * (theta + 1) * np.power(t, -theta - 2)

    def deriv_inverse(self, phi, theta):
        return -(1 + phi) ** (-(theta + 1) / theta) / theta

    def deriv2_inverse(self, phi, theta):
        return (theta + 1) * (1 + phi) ** (-1 / theta - 2) / theta ** 2

    def deriv3_inverse(self, phi, theta):
        th = theta
        d3 = -((1 + th) * (1 + 2 * th) / th ** 3 * (1 + phi) ** (-1 / th - 3))
        return d3

    def deriv4_inverse(self, phi, theta):
        th = theta
        d4 = (1 + th) * (1 + 2 * th) * (1 + 3 * th) / th ** 4 * (1 + phi) ** (-1 / th - 4)
        return d4

    def derivk_inverse(self, k, phi, theta):
        thi = 1 / theta
        d4 = (-1) ** k * gamma(k + thi) / gamma(thi) * (1 + phi) ** (-(k + thi))
        return d4

    def is_completly_monotonic(self, theta):
        return theta > 0