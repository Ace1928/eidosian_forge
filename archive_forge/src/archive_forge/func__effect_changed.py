import unittest
from traits.api import HasTraits, Str, Instance, Any
def _effect_changed(self, obj, name, old, new):
    self.test.events_delivered.append('Baz._effect_changed')