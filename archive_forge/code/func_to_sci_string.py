import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def to_sci_string(self, a):
    """Converts a number to a string, using scientific notation.

        The operation is not affected by the context.
        """
    a = _convert_other(a, raiseit=True)
    return a.__str__(context=self)