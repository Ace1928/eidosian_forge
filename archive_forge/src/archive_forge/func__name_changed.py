import unittest
from traits.api import HasTraits, Str, Undefined, ReadOnly, Float
def _name_changed(self):
    if self.original_name is Undefined:
        self.original_name = self.name