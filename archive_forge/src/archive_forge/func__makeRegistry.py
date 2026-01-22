import unittest
from zope.interface.tests import OptimizationTestMixin
def _makeRegistry(self, *provided):

    class Registry:

        def __init__(self, provided):
            self._provided = provided
            self.ro = []
    return Registry(provided)