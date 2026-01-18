from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_on_callable_objects(self):

    class Foo(object):

        def __call__(self, a):
            pass
    self.assertEqual(self.signature(Foo()), ((('a', Ellipsis, Ellipsis, 'positional_or_keyword'),), Ellipsis))

    class Spam(object):
        pass
    with self.assertRaisesRegex(TypeError, 'is not a callable object'):
        inspect.signature(Spam())

    class Bar(Spam, Foo):
        pass
    self.assertEqual(self.signature(Bar()), ((('a', Ellipsis, Ellipsis, 'positional_or_keyword'),), Ellipsis))

    class ToFail(object):
        __call__ = type
    with self.assertRaisesRegex(ValueError, 'not supported by signature'):
        inspect.signature(ToFail())
    if sys.version_info[0] < 3:
        return

    class Wrapped(object):
        pass
    Wrapped.__wrapped__ = lambda a: None
    self.assertEqual(self.signature(Wrapped), ((('a', Ellipsis, Ellipsis, 'positional_or_keyword'),), Ellipsis))