import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class IRange(Interface):
    min = Attribute('Lower bound')
    max = Attribute('Upper bound')

    @invariant
    def range_invariant(ob):
        if ob.max < ob.min:
            raise Invalid('max < min')