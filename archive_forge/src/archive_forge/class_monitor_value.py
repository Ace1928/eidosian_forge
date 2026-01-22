from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
class monitor_value(int_value):

    def init(self):
        self.on_trait_change(self.on_value_change, 'value')

    def on_value_change(self, object, trait_name, old, new):
        pass