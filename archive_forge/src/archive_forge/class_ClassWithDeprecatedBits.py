import unittest
from traits.testing.api import UnittestTools
from traits.util.api import deprecated
class ClassWithDeprecatedBits(object):

    @deprecated('bits are deprecated; use bytes')
    def bits(self):
        return 42

    @deprecated('bytes are deprecated too. Use base 10.')
    def bytes(self, required_arg, *args, **kwargs):
        return (required_arg, args, kwargs)