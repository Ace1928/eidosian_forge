from . import matrix
from . import homology
from .polynomial import Polynomial
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase, processFileDispatch, processMagmaFile
from . import utilities
from string import Template
import signal
import re
import os
import sys
from urllib.request import Request, urlopen
from urllib.request import quote as urlquote
from urllib.error import HTTPError
def _canonical_representative_to_polynomial_substituition(canonical_representative, order_of_u):
    result = {}
    for var1, signed_var2 in canonical_representative.items():
        sign, power, var2 = signed_var2
        if not var1 == var2:
            if order_of_u == 2:
                u = Polynomial.constant_polynomial(-1)
            else:
                u = Polynomial.from_variable_name('u')
            sign_and_power = Polynomial.constant_polynomial(sign) * u ** (power % order_of_u)
            if var2 == 1:
                result[var1] = sign_and_power
            else:
                result[var1] = sign_and_power * Polynomial.from_variable_name(var2)
    return result