from decimal import Decimal
import math
import numbers
import operator
import re
import sys
def _divmod(a, b):
    """(a // b, a % b)"""
    da, db = (a.denominator, b.denominator)
    div, n_mod = divmod(a.numerator * db, da * b.numerator)
    return (div, Fraction(n_mod, da * db))