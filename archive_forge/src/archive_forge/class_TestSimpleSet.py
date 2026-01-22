import sys
from breezy import tests
from breezy.tests import features
class TestSimpleSet(tests.TestCase):
    _test_needs_features = [compiled_simpleset_feature]
    module = _simple_set_pyx

    def assertFillState(self, used, fill, mask, obj):
        self.assertEqual((used, fill, mask), (obj.used, obj.fill, obj.mask))

    def assertLookup(self, offset, value, obj, key):
        self.assertEqual((offset, value), obj._test_lookup(key))

    def assertRefcount(self, count, obj):
        """Assert that the refcount for obj is what we expect.

        Note that this automatically adjusts for the fact that calling
        assertRefcount actually creates a new pointer, as does calling
        sys.getrefcount. So pass the expected value *before* the call.
        """
        if sys.version_info < (3, 11):
            self.assertEqual(count, sys.getrefcount(obj) - 3)
        else:
            self.assertEqual(count, sys.getrefcount(obj) - 2)

    def test_initial(self):
        obj = self.module.SimpleSet()
        self.assertEqual(0, len(obj))
        self.assertFillState(0, 0, 1023, obj)

    def test__lookup(self):
        obj = self.module.SimpleSet()
        self.assertLookup(643, '<null>', obj, _Hashable(643))
        self.assertLookup(643, '<null>', obj, _Hashable(643 + 1024))
        self.assertLookup(643, '<null>', obj, _Hashable(643 + 50 * 1024))

    def test__lookup_collision(self):
        obj = self.module.SimpleSet()
        k1 = _Hashable(643)
        k2 = _Hashable(643 + 1024)
        self.assertLookup(643, '<null>', obj, k1)
        self.assertLookup(643, '<null>', obj, k2)
        obj.add(k1)
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(644, '<null>', obj, k2)

    def test__lookup_after_resize(self):
        obj = self.module.SimpleSet()
        k1 = _Hashable(643)
        k2 = _Hashable(643 + 1024)
        obj.add(k1)
        obj.add(k2)
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(644, k2, obj, k2)
        obj._py_resize(2047)
        self.assertEqual(2048, obj.mask + 1)
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(643 + 1024, k2, obj, k2)
        obj._py_resize(1023)
        self.assertEqual(1024, obj.mask + 1)
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(644, k2, obj, k2)

    def test_get_set_del_with_collisions(self):
        obj = self.module.SimpleSet()
        h1 = 643
        h2 = 643 + 1024
        h3 = 643 + 1024 * 50
        h4 = 643 + 1024 * 25
        h5 = 644
        h6 = 644 + 1024
        k1 = _Hashable(h1)
        k2 = _Hashable(h2)
        k3 = _Hashable(h3)
        k4 = _Hashable(h4)
        k5 = _Hashable(h5)
        k6 = _Hashable(h6)
        self.assertLookup(643, '<null>', obj, k1)
        self.assertLookup(643, '<null>', obj, k2)
        self.assertLookup(643, '<null>', obj, k3)
        self.assertLookup(643, '<null>', obj, k4)
        self.assertLookup(644, '<null>', obj, k5)
        self.assertLookup(644, '<null>', obj, k6)
        obj.add(k1)
        self.assertIn(k1, obj)
        self.assertNotIn(k2, obj)
        self.assertNotIn(k3, obj)
        self.assertNotIn(k4, obj)
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(644, '<null>', obj, k2)
        self.assertLookup(644, '<null>', obj, k3)
        self.assertLookup(644, '<null>', obj, k4)
        self.assertLookup(644, '<null>', obj, k5)
        self.assertLookup(644, '<null>', obj, k6)
        self.assertIs(k1, obj[k1])
        self.assertIs(k2, obj.add(k2))
        self.assertIs(k2, obj[k2])
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(644, k2, obj, k2)
        self.assertLookup(646, '<null>', obj, k3)
        self.assertLookup(646, '<null>', obj, k4)
        self.assertLookup(645, '<null>', obj, k5)
        self.assertLookup(645, '<null>', obj, k6)
        self.assertLookup(643, k1, obj, _Hashable(h1))
        self.assertLookup(644, k2, obj, _Hashable(h2))
        self.assertLookup(646, '<null>', obj, _Hashable(h3))
        self.assertLookup(646, '<null>', obj, _Hashable(h4))
        self.assertLookup(645, '<null>', obj, _Hashable(h5))
        self.assertLookup(645, '<null>', obj, _Hashable(h6))
        obj.add(k3)
        self.assertIs(k3, obj[k3])
        self.assertIn(k1, obj)
        self.assertIn(k2, obj)
        self.assertIn(k3, obj)
        self.assertNotIn(k4, obj)
        obj.discard(k1)
        self.assertLookup(643, '<dummy>', obj, k1)
        self.assertLookup(644, k2, obj, k2)
        self.assertLookup(646, k3, obj, k3)
        self.assertLookup(643, '<dummy>', obj, k4)
        self.assertNotIn(k1, obj)
        self.assertIn(k2, obj)
        self.assertIn(k3, obj)
        self.assertNotIn(k4, obj)

    def test_add(self):
        obj = self.module.SimpleSet()
        self.assertFillState(0, 0, 1023, obj)
        k1 = tuple(['foo'])
        self.assertRefcount(1, k1)
        self.assertIs(k1, obj.add(k1))
        self.assertFillState(1, 1, 1023, obj)
        self.assertRefcount(2, k1)
        ktest = obj[k1]
        self.assertRefcount(3, k1)
        self.assertIs(k1, ktest)
        del ktest
        self.assertRefcount(2, k1)
        k2 = tuple(['foo'])
        self.assertRefcount(1, k2)
        self.assertIsNot(k1, k2)
        self.assertIs(k1, obj.add(k2))
        self.assertFillState(1, 1, 1023, obj)
        self.assertRefcount(2, k1)
        self.assertRefcount(1, k2)
        self.assertIs(k1, obj[k1])
        self.assertIs(k1, obj[k2])
        self.assertRefcount(2, k1)
        self.assertRefcount(1, k2)
        obj.discard(k1)
        self.assertFillState(0, 1, 1023, obj)
        self.assertRefcount(1, k1)
        k3 = tuple(['bar'])
        self.assertRefcount(1, k3)
        self.assertIs(k3, obj.add(k3))
        self.assertFillState(1, 2, 1023, obj)
        self.assertRefcount(2, k3)
        self.assertIs(k2, obj.add(k2))
        self.assertFillState(2, 2, 1023, obj)
        self.assertRefcount(1, k1)
        self.assertRefcount(2, k2)
        self.assertRefcount(2, k3)

    def test_discard(self):
        obj = self.module.SimpleSet()
        k1 = tuple(['foo'])
        k2 = tuple(['foo'])
        k3 = tuple(['bar'])
        self.assertRefcount(1, k1)
        self.assertRefcount(1, k2)
        self.assertRefcount(1, k3)
        obj.add(k1)
        self.assertRefcount(2, k1)
        self.assertEqual(0, obj.discard(k3))
        self.assertRefcount(1, k3)
        obj.add(k3)
        self.assertRefcount(2, k3)
        self.assertEqual(1, obj.discard(k3))
        self.assertRefcount(1, k3)

    def test__resize(self):
        obj = self.module.SimpleSet()
        k1 = _Hashable(501)
        k2 = _Hashable(591)
        k3 = _Hashable(2051)
        obj.add(k1)
        obj.add(k2)
        obj.add(k3)
        obj.discard(k2)
        self.assertFillState(2, 3, 1023, obj)
        self.assertEqual(1024, obj._py_resize(500))
        self.assertFillState(2, 2, 1023, obj)
        obj.add(k2)
        obj.discard(k3)
        self.assertFillState(2, 3, 1023, obj)
        self.assertEqual(4096, obj._py_resize(4095))
        self.assertFillState(2, 2, 4095, obj)
        self.assertIn(k1, obj)
        self.assertIn(k2, obj)
        self.assertNotIn(k3, obj)
        obj.add(k2)
        self.assertIn(k2, obj)
        obj.discard(k2)
        self.assertEqual((591, '<dummy>'), obj._test_lookup(k2))
        self.assertFillState(1, 2, 4095, obj)
        self.assertEqual(2048, obj._py_resize(1024))
        self.assertFillState(1, 1, 2047, obj)
        self.assertEqual((591, '<null>'), obj._test_lookup(k2))

    def test_second_hash_failure(self):
        obj = self.module.SimpleSet()
        k1 = _BadSecondHash(200)
        k2 = _Hashable(200)
        obj.add(k1)
        self.assertFalse(k1._first)
        self.assertRaises(ValueError, obj.add, k2)

    def test_richcompare_failure(self):
        obj = self.module.SimpleSet()
        k1 = _Hashable(200)
        k2 = _BadCompare(200)
        obj.add(k1)
        self.assertRaises(RuntimeError, obj.add, k2)

    def test_richcompare_not_implemented(self):
        obj = self.module.SimpleSet()
        k1 = _NoImplementCompare(200)
        k2 = _NoImplementCompare(200)
        self.assertLookup(200, '<null>', obj, k1)
        self.assertLookup(200, '<null>', obj, k2)
        self.assertIs(k1, obj.add(k1))
        self.assertLookup(200, k1, obj, k1)
        self.assertLookup(201, '<null>', obj, k2)
        self.assertIs(k2, obj.add(k2))
        self.assertIs(k1, obj[k1])

    def test_add_and_remove_lots_of_items(self):
        obj = self.module.SimpleSet()
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890'
        for i in chars:
            for j in chars:
                k = (i, j)
                obj.add(k)
        num = len(chars) * len(chars)
        self.assertFillState(num, num, 8191, obj)
        for i in chars:
            for j in chars:
                k = (i, j)
                obj.discard(k)
        self.assertFillState(0, obj.fill, 1023, obj)
        self.assertTrue(obj.fill < 1024 / 5)

    def test__iter__(self):
        obj = self.module.SimpleSet()
        k1 = ('1',)
        k2 = ('1', '2')
        k3 = ('3', '4')
        obj.add(k1)
        obj.add(k2)
        obj.add(k3)
        all = set()
        for key in obj:
            all.add(key)
        self.assertEqual(sorted([k1, k2, k3]), sorted(all))
        iterator = iter(obj)
        self.assertIn(next(iterator), all)
        obj.add(('foo',))
        self.assertRaises(RuntimeError, next, iterator)
        obj.discard(k2)
        self.assertRaises(RuntimeError, next, iterator)

    def test__sizeof__(self):
        obj = self.module.SimpleSet()
        self.assertTrue(obj.__sizeof__() > 4096)