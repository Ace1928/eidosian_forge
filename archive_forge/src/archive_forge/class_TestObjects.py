from h5py import _objects as o
from .common import TestCase
class TestObjects(TestCase):

    def test_invalid(self):
        oid = o.ObjectID(0)
        del oid
        oid = o.ObjectID(1)
        del oid

    def test_equality(self):
        oid1 = o.ObjectID(42)
        oid2 = o.ObjectID(42)
        oid3 = o.ObjectID(43)
        self.assertEqual(oid1, oid2)
        self.assertNotEqual(oid1, oid3)

    def test_hash(self):
        oid = o.ObjectID(42)
        with self.assertRaises(TypeError):
            hash(oid)