import unittest
import warnings
from traits.api import (
def test_rich_compare_with_cached_property(self):

    class Model(HasTraits):
        value = Property(depends_on='name')
        name = Str(comparison_mode=ComparisonMode.none)

        @cached_property
        def _get_value(self):
            return self.trait_names
    instance = Model()
    events = []
    instance.on_trait_change(lambda: events.append(None), 'value')
    instance.name = 'A'
    events.clear()
    instance.name = 'A'
    self.assertEqual(len(events), 1)