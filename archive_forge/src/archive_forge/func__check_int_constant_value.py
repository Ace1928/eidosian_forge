import sys, os
import types
from . import model
from .error import VerificationError
def _check_int_constant_value(self, name, value):
    prnt = self._prnt
    if value <= 0:
        prnt('  if ((%s) > 0 || (long)(%s) != %dL) {' % (name, name, value))
    else:
        prnt('  if ((%s) <= 0 || (unsigned long)(%s) != %dUL) {' % (name, name, value))
    prnt('    char buf[64];')
    prnt('    if ((%s) <= 0)' % name)
    prnt('        sprintf(buf, "%%ld", (long)(%s));' % name)
    prnt('    else')
    prnt('        sprintf(buf, "%%lu", (unsigned long)(%s));' % name)
    prnt('    sprintf(out_error, "%s has the real value %s, not %s",')
    prnt('            "%s", buf, "%d");' % (name[:100], value))
    prnt('    return -1;')
    prnt('  }')