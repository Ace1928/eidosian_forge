import unittest
import warnings
from traits.api import (
def test_rich_first_assignment(self):
    rich = RichCompare()
    rich.on_trait_change(self.bar_changed, 'bar')
    self.reset_change_tracker()
    default_value = rich.bar
    rich.bar = self.a
    self.check_tracker(rich, 'bar', default_value, self.a, 1)