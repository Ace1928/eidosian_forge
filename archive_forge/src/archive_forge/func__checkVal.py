from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
def _checkVal(self, name, v, allowed):
    if v not in allowed:
        raise ValueError('%s attribute %s is invalid %r\nnot in allowed %r' % (self.__class__.__name__, name, v, allowed))
    return v