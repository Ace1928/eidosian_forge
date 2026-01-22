import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def Etiny(self):
    """Returns Etiny (= Emin - prec + 1)"""
    return int(self.Emin - self.prec + 1)