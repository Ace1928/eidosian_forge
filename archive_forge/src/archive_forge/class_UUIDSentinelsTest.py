import datetime
from oslotest import base as test_base
from oslo_utils import fixture
from oslo_utils.fixture import keystoneidsentinel as keystids
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from oslo_utils import uuidutils
class UUIDSentinelsTest(test_base.BaseTestCase):

    def test_different_sentinel(self):
        uuid1 = uuids.foobar
        uuid2 = uuids.barfoo
        self.assertNotEqual(uuid1, uuid2)
        keystid1 = keystids.foobar
        keystid2 = keystids.barfoo
        self.assertNotEqual(keystid1, keystid2)

    def test_returns_uuid(self):
        self.assertTrue(uuidutils.is_uuid_like(uuids.foo))
        self.assertTrue(uuidutils.is_uuid_like(keystids.foo))

    def test_returns_string(self):
        self.assertIsInstance(uuids.foo, str)
        self.assertIsInstance(keystids.foo, str)

    def test_with_underline_prefix(self):
        ex = self.assertRaises(AttributeError, getattr, uuids, '_foo')
        self.assertIn('Sentinels must not start with _', str(ex))
        ex = self.assertRaises(AttributeError, getattr, keystids, '_foo')
        self.assertIn('Sentinels must not start with _', str(ex))