import unittest
def test_w_bases(self):

    class Foo:
        pass

    class Bar(Foo):
        pass
    self.assertEqual(self._callFUT(Bar), [Bar, Foo, object])