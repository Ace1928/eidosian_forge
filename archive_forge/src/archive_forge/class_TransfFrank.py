import warnings
import numpy as np
from scipy.special import expm1, gamma
class TransfFrank(Transforms):

    def evaluate(self, t, theta):
        t = np.asarray(t)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            val = -(np.log(-expm1(-theta * t)) - np.log(-expm1(-theta)))
        return val

    def inverse(self, phi, theta):
        phi = np.asarray(phi)
        return -np.log1p(np.exp(-phi) * expm1(-theta)) / theta

    def deriv(self, t, theta):
        t = np.asarray(t)
        tmp = np.exp(-t * theta)
        return -theta * tmp / (tmp - 1)

    def deriv2(self, t, theta):
        t = np.asarray(t)
        tmp = np.exp(theta * t)
        d2 = -theta ** 2 * tmp / (tmp - 1) ** 2
        return d2

    def deriv2_inverse(self, phi, theta):
        et = np.exp(theta)
        ept = np.exp(phi + theta)
        d2 = (et - 1) * ept / (theta * (ept - et + 1) ** 2)
        return d2

    def deriv3_inverse(self, phi, theta):
        et = np.exp(theta)
        ept = np.exp(phi + theta)
        d3 = -((et - 1) * ept * (ept + et - 1) / (theta * (ept - et + 1) ** 3))
        return d3

    def deriv4_inverse(self, phi, theta):
        et = np.exp(theta)
        ept = np.exp(phi + theta)
        p = phi
        b = theta
        d4 = (et - 1) * ept * (-4 * ept + np.exp(2 * (p + b)) + 4 * np.exp(p + 2 * b) - 2 * et + np.exp(2 * b) + 1) / (b * (ept - et + 1) ** 4)
        return d4

    def is_completly_monotonic(self, theta):
        return theta > 0 & theta < 1