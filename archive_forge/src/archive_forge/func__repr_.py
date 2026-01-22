from .sage_helper import _within_sage
from .pari import *
import re
def _repr_(self):
    return 'SnapPy Numbers with %s bits precision' % self._precision