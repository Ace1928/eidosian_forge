from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
class delegate_3_value(delegate_value):

    def init(self):
        delegate = delegate_2_value()
        delegate.init()
        self.delegate = delegate