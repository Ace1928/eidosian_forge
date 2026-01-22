import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def Etop(self):
    """Returns maximum exponent (= Emax - prec + 1)"""
    return int(self.Emax - self.prec + 1)