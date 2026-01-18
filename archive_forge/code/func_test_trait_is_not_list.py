import unittest
from traits.api import (
from traits.observation.api import (
def test_trait_is_not_list(self):
    team = Team()
    team.observe(lambda e: None, trait('leader').list_items())
    person = Person()
    with self.assertRaises(ValueError) as exception_cm:
        team.leader = person
    self.assertIn('Expected a TraitList to be observed', str(exception_cm.exception))