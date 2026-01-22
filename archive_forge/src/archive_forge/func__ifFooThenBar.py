import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def _ifFooThenBar(obj):
    from zope.interface.exceptions import Invalid
    if getattr(obj, 'foo', None) and (not getattr(obj, 'bar', None)):
        raise Invalid('If Foo, then Bar!')