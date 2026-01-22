import sys
import unittest
import sys
class Test_determineMetaclass(unittest.TestCase):

    def _callFUT(self, *args, **kw):
        from zope.interface.advice import determineMetaclass
        return determineMetaclass(*args, **kw)

    def test_empty_w_explicit_metatype(self):

        class Meta(type):
            pass
        self.assertEqual(self._callFUT((), Meta), Meta)

    def test_single(self):

        class Meta(type):
            pass
        self.assertEqual(self._callFUT((Meta,)), type)

    def test_meta_of_class(self):

        class Metameta(type):
            pass

        class Meta(type, metaclass=Metameta):
            pass
        self.assertEqual(self._callFUT((Meta, type)), Metameta)

    def test_multiple_in_hierarchy_py3k(self):

        class Meta_A(type):
            pass

        class Meta_B(Meta_A):
            pass

        class A(type, metaclass=Meta_A):
            pass

        class B(type, metaclass=Meta_B):
            pass
        self.assertEqual(self._callFUT((A, B)), Meta_B)

    def test_multiple_not_in_hierarchy_py3k(self):

        class Meta_A(type):
            pass

        class Meta_B(type):
            pass

        class A(type, metaclass=Meta_A):
            pass

        class B(type, metaclass=Meta_B):
            pass
        self.assertRaises(TypeError, self._callFUT, (A, B))