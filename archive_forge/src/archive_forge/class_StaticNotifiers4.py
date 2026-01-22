import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
class StaticNotifiers4(HasTraits):
    ok = Float

    def _ok_changed(self, name, old, new):
        if not hasattr(self, 'calls'):
            self.calls = []
        self.calls.append((name, old, new))
    fail = Float

    def _fail_changed(self, name, old, new):
        raise Exception('error')