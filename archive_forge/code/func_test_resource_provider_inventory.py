import uuid
from openstack.placement.v1 import trait as _trait
from openstack.tests.functional import base
def test_resource_provider_inventory(self):
    traits = list(self.operator_cloud.placement.traits())
    self.assertIsInstance(traits[0], _trait.Trait)
    self.assertIn(self.trait.name, {x.id for x in traits})
    trait = self.operator_cloud.placement.get_trait(self.trait)
    self.assertIsInstance(trait, _trait.Trait)
    self.assertEqual(self.trait_name, trait.id)
    trait = self.operator_cloud.placement.get_trait(self.trait_name)
    self.assertIsInstance(trait, _trait.Trait)
    self.assertEqual(self.trait_name, trait.id)