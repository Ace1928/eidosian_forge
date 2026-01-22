import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class ExtendsRoot(Root1, Root2):
    __implemented__ = impl_extends_root