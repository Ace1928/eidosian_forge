import uuid
from osc_placement.tests.functional import base
def test_set_known_and_unknown_traits(self):
    self.trait_create(TRAIT)
    rp = self.resource_provider_create()
    self.assertCommandFailed('No such trait', self.resource_provider_trait_set, rp['uuid'], TRAIT, TRAIT + '1')
    self.assertEqual([], self.resource_provider_trait_list(rp['uuid']))