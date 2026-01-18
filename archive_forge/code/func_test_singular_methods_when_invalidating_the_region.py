import uuid
from dogpile.cache import api as dogpile
from dogpile.cache.backends import memory
from oslo_config import fixture as config_fixture
from keystone.common import cache
import keystone.conf
from keystone.tests import unit
def test_singular_methods_when_invalidating_the_region(self):
    key = uuid.uuid4().hex
    value = uuid.uuid4().hex
    self.assertIsInstance(self.region0.get(key), dogpile.NoValue)
    self.region0.set(key, value)
    self.assertEqual(value, self.region0.get(key))
    self.region1.invalidate()
    self.assertIsInstance(self.region0.get(key), dogpile.NoValue)