from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
class delegate_value(HasTraits, new_style_value):
    value = DelegatesTo('delegate')
    delegate = Any

    def init(self):
        self.delegate = int_value()