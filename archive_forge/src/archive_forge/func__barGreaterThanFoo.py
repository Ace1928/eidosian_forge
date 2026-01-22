import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def _barGreaterThanFoo(obj):
    from zope.interface.exceptions import Invalid
    foo = getattr(obj, 'foo', None)
    bar = getattr(obj, 'bar', None)
    if foo is not None and isinstance(foo, type(bar)):
        if not bar > foo:
            raise Invalid('Please, Boo MUST be greater than Foo!')