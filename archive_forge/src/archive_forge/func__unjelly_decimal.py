import copy
import datetime
import decimal
import types
import warnings
from functools import reduce
from zope.interface import implementer
from incremental import Version
from twisted.persisted.crefutil import (
from twisted.python.compat import nativeString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import namedAny, namedObject, qual
from twisted.spread.interfaces import IJellyable, IUnjellyable
def _unjelly_decimal(self, exp):
    """
        Unjelly decimal objects.
        """
    value = exp[0]
    exponent = exp[1]
    if value < 0:
        sign = 1
    else:
        sign = 0
    guts = decimal.Decimal(value).as_tuple()[1]
    return decimal.Decimal((sign, guts, exponent))