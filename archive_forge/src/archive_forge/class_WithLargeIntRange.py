import unittest
from traits.api import HasTraits, Int, List, Range, Str, TraitError, Tuple
class WithLargeIntRange(HasTraits):
    r = Range(0, 1000)
    r_copied_on_change = Str
    _changed_handler_calls = Int

    def _r_changed(self, old, new):
        self._changed_handler_calls += 1
        self.r_copied_on_change = str(self.r)
        if self.r > 100:
            self.r = 0