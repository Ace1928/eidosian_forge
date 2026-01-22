import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def PolishPostalCode(*args, **kw):
    return DelimitedDigitsPostalCode([2, 3], '-', *args, **kw)