import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def _validateMod10(self, s):
    """Check string with the mod 10 algorithm (aka "Luhn formula")."""
    checksum, factor = (0, 1)
    for c in reversed(s):
        for c in str(factor * int(c)):
            checksum += int(c)
        factor = 3 - factor
    return checksum % 10 == 0