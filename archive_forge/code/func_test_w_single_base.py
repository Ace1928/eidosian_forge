import unittest
def test_w_single_base(self):

    class Foo:
        pass
    self.assertEqual(self._callFUT(Foo), [Foo, object])