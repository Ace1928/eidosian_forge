from unittest import TestCase
import simplejson as json
from decimal import Decimal
class AlternateFloat(float):

    def __repr__(self):
        return 'invalid json'
    __str__ = __repr__