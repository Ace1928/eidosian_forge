import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def GermanPostalCode(*args, **kw):
    return DelimitedDigitsPostalCode(5, None, *args, **kw)