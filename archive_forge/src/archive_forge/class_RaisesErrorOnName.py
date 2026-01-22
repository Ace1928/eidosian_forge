import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class RaisesErrorOnName(RaisesErrorOnMissing):

    def __init__(self):
        self.__module__ = 'foo'