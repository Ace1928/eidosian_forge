import calendar
from typing import Any, Optional, Tuple
class VerifiedPassword(String):
    """A string that should be obscured when input and needs verification."""

    def coerce(self, vals):
        if len(vals) != 2 or vals[0] != vals[1]:
            raise InputError('Please enter the same password twice.')
        s = str(vals[0])
        if len(s) < self.min:
            raise InputError('Value must be at least %s characters long' % self.min)
        if self.max is not None and len(s) > self.max:
            raise InputError('Value must be at most %s characters long' % self.max)
        return s