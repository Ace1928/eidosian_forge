import numpy as np
from scipy import stats
from statsmodels.tools.numdiff import _approx_fprime_cs_scalar, approx_hess
class AsymNegLogistic(PickandDependence):
    """asymmetric negative logistic model of Joe 1990

    special case:  a1=a2=1 : symmetric negative logistic of Galambos 1978

    restrictions:
     - theta in (0,inf)
     - a1, a2 in (0,1]
    """
    k_args = 3

    def _check_args(self, a1, a2, theta):
        condth = theta > 0
        conda1 = a1 > 0 and a1 <= 1
        conda2 = a2 > 0 and a2 <= 1
        return condth and conda1 and conda2

    def evaluate(self, t, a1, a2, theta):
        a1, a2 = (a2, a1)
        transf = 1 - ((a1 * (1 - t)) ** (-1.0 / theta) + (a2 * t) ** (-1.0 / theta)) ** (-theta)
        return transf

    def deriv(self, t, a1, a2, theta):
        a1, a2 = (a2, a1)
        m1 = -1 / theta
        m2 = m1 - 1
        d1 = (a1 ** m1 * (1 - t) ** m2 - a2 ** m1 * t ** m2) * ((a1 * (1 - t)) ** m1 + (a2 * t) ** m1) ** (-theta - 1)
        return d1

    def deriv2(self, t, a1, a2, theta):
        b = theta
        a1, a2 = (a2, a1)
        a1tp = (a1 * (1 - t)) ** (1 / b)
        a2tp = (a2 * t) ** (1 / b)
        a1tn = (a1 * (1 - t)) ** (-1 / b)
        a2tn = (a2 * t) ** (-1 / b)
        t1 = (b + 1) * a2tp * a1tp * (a1tn + a2tn) ** (-b)
        t2 = b * (1 - t) ** 2 * t ** 2 * (a1tp + a2tp) ** 2
        d2 = t1 / t2
        return d2