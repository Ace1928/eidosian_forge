import unittest
from traits.api import (
from traits.observation.api import (
def test_shared_instance_but_different_target(self):
    potato = Potato()
    potato_bag = PotatoBag(potatos=[potato])
    crate1 = Crate(potato_bags=[potato_bag])
    crate2 = Crate(potato_bags=[potato_bag])
    events = []
    handler = events.append
    crate1.observe(handler, 'potato_bags:items:potatos:items:name')
    crate2.observe(handler, 'potato_bags:items:potatos:items:name')
    potato.name = 'King Edward'
    self.assertEqual(len(events), 2)