import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class RaisesErrorOnMissing:
    Exc = AttributeError

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            exc = RaisesErrorOnMissing.Exc
            raise exc(name)