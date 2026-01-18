import unittest2 as unittest
import doctest
import sys
import funcsigs as inspect
def test_unbound_method(self):
    self_kind = 'positional_or_keyword'

    class Test(object):

        def method(self):
            pass

        def method_with_args(self, a):
            pass

        def method_with_varargs(*args):
            pass
    self.assertEqual(self.signature(Test.method), ((('self', Ellipsis, Ellipsis, self_kind),), Ellipsis))
    self.assertEqual(self.signature(Test.method_with_args), ((('self', Ellipsis, Ellipsis, self_kind), ('a', Ellipsis, Ellipsis, 'positional_or_keyword')), Ellipsis))
    self.assertEqual(self.signature(Test.method_with_varargs), ((('args', Ellipsis, Ellipsis, 'var_positional'),), Ellipsis))