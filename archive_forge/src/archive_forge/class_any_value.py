from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
class any_value(HasTraits, new_style_value):
    value = Any