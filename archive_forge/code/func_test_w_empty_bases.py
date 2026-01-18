import unittest
def test_w_empty_bases(self):

    class Foo:
        pass
    foo = Foo()
    foo.__bases__ = ()
    self.assertEqual(self._callFUT(foo), [foo])