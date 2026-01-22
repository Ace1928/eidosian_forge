import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class AllowsAnyComparison(RaisesErrorOnMissing):

    def __eq__(self, other):
        return True
    __lt__ = __eq__
    __le__ = __eq__
    __gt__ = __eq__
    __ge__ = __eq__
    __ne__ = __eq__