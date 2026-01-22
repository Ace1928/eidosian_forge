import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def _ok_changed(self, name, old, new):
    if not hasattr(self, 'calls'):
        self.calls = []
    self.calls.append((name, old, new))