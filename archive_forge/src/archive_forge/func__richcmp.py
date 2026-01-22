from decimal import Decimal
import math
import numbers
import operator
import re
import sys
def _richcmp(self, other, op):
    """Helper for comparison operators, for internal use only.

        Implement comparison between a Rational instance `self`, and
        either another Rational instance or a float `other`.  If
        `other` is not a Rational instance or a float, return
        NotImplemented. `op` should be one of the six standard
        comparison operators.

        """
    if isinstance(other, numbers.Rational):
        return op(self._numerator * other.denominator, self._denominator * other.numerator)
    if isinstance(other, float):
        if math.isnan(other) or math.isinf(other):
            return op(0.0, other)
        else:
            return op(self, self.from_float(other))
    else:
        return NotImplemented