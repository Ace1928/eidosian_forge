from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
class int_value(any_value):
    value = Int