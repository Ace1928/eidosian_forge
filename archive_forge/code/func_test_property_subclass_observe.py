import unittest
from traits.api import (
from traits.observation.api import (
def test_property_subclass_observe(self):

    class Base(HasTraits):
        bar = Int()
        foo = Property(Int(), observe='bar')

        def _get_foo(self):
            return self.bar

    class Derived(Base):
        pass
    events = []
    obj = Derived(bar=3)
    obj.observe(events.append, 'foo')
    self.assertEqual(len(events), 0)
    obj.bar = 5
    self.assertEqual(len(events), 1)