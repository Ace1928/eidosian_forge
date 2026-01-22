import unittest
from zope.interface.tests import OptimizationTestMixin
def _factory2(context):
    _called.setdefault('_factory2', []).append(context)
    return _exp2