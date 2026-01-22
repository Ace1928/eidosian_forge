import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def _strict_call(self, value):
    try:
        new_value = self.func(value)
        if self.func is int:
            try:
                np.array(value, dtype=self.type)
            except OverflowError:
                raise ValueError
        return new_value
    except ValueError:
        if value.strip() in self.missing_values:
            if not self._status:
                self._checked = False
            return self.default
        raise ValueError("Cannot convert string '%s'" % value)