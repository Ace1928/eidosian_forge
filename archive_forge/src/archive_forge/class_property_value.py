from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
class property_value(new_style_value):

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value
    value = property(get_value, set_value)