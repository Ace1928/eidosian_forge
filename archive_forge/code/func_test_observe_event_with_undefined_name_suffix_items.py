import unittest
from traits.api import (
from traits.observation.api import (
def test_observe_event_with_undefined_name_suffix_items(self):
    app = Application()

    def dummy_handler():
        pass
    app.on_trait_change(dummy_handler, 'i_am_undefined_with_items')
    self.assertIsNotNone(app._trait('i_am_undefined_with_items', 0))
    self.assertNotIn('i_am_undefined_with_items', app.traits())
    events = []
    app.observe(events.append, 'i_am_undefined_with_items')
    app.trait_property_changed('i_am_undefined_with_items', 1, 2)
    self.assertEqual(len(events), 1)