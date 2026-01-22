import unittest
from traits.api import (
from traits.observation.api import (
class HasVariousTraits(HasTraits):
    trait_change_callback = Any()
    foo = Int(16)
    bar = Str('off')
    updated = Event(Bool)

    @observe('*')
    def _record_trait_change(self, event):
        callback = self.trait_change_callback
        if callback is not None:
            callback(event)