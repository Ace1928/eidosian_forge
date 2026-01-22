import operator
import math
import cmath
def _erfc_mid(x):
    return exp(-x * x) * _polyval(_erfc_coeff_P, x) / _polyval(_erfc_coeff_Q, x)