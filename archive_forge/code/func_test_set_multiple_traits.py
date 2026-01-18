import uuid
from osc_placement.tests.functional import base
def test_set_multiple_traits(self):
    self.trait_create(TRAIT + '1')
    self.trait_create(TRAIT + '2')
    rp = self.resource_provider_create()
    self.resource_provider_trait_set(rp['uuid'], TRAIT + '1', TRAIT + '2')
    traits = {t['name'] for t in self.resource_provider_trait_list(rp['uuid'])}
    self.assertEqual(2, len(traits))