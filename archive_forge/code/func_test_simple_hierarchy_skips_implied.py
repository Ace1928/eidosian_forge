import sys
import unittest
import sys
def test_simple_hierarchy_skips_implied(self):

    class A:
        pass

    class B(A):
        pass

    class C(B):
        pass

    class D:
        pass
    self.assertEqual(self._callFUT([A, B, C]), [C])
    self.assertEqual(self._callFUT([A, C]), [C])
    self.assertEqual(self._callFUT([B, C]), [C])
    self.assertEqual(self._callFUT([A, B]), [B])
    self.assertEqual(self._callFUT([D, B, D]), [B, D])