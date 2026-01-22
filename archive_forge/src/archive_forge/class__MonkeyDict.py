import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class _MonkeyDict:

    def __init__(self, module, attrname, **kw):
        self.module = module
        self.target = getattr(module, attrname)
        self.to_restore = self.target.copy()
        self.target.clear()
        self.target.update(kw)

    def __enter__(self):
        return self.target

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.target.clear()
        self.target.update(self.to_restore)