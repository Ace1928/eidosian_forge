import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_rich_compare_false(self):
    with warnings.catch_warnings(record=True) as warn_msgs:
        warnings.simplefilter('always', DeprecationWarning)

        class OldRichCompare(HasTraits):
            bar = Trait(rich_compare=False)
    self.assertEqual(len(warn_msgs), 1)
    warn_msg = warn_msgs[0]
    self.assertIs(warn_msg.category, DeprecationWarning)
    self.assertIn("'rich_compare' metadata has been deprecated", str(warn_msg.message))
    _, _, this_module = __name__.rpartition('.')
    self.assertIn(this_module, warn_msg.filename)
    old_compare = OldRichCompare()
    events = []
    old_compare.on_trait_change(lambda: events.append(None), 'bar')
    some_list = [1, 2, 3]
    self.assertEqual(len(events), 0)
    old_compare.bar = some_list
    self.assertEqual(len(events), 1)
    old_compare.bar = some_list
    self.assertEqual(len(events), 1)
    old_compare.bar = [1, 2, 3]
    self.assertEqual(len(events), 2)
    old_compare.bar = [4, 5, 6]
    self.assertEqual(len(events), 3)