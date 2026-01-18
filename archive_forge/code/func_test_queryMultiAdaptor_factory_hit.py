import unittest
from zope.interface.tests import OptimizationTestMixin
def test_queryMultiAdaptor_factory_hit(self):
    from zope.interface.declarations import implementer
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))

    @implementer(IFoo)
    class Foo:
        pass
    foo = Foo()
    registry = self._makeRegistry(IFoo, IBar)
    subr = self._makeSubregistry()
    _expected = object()
    _called_with = []

    def _factory(context):
        _called_with.append(context)
        return _expected
    subr._adapters = [{}, {IFoo: {IBar: {'': _factory}}}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    alb.lookup = alb._uncached_lookup
    subr._v_lookup = alb
    _default = object()
    result = alb.queryMultiAdapter((foo,), IBar, default=_default)
    self.assertIs(result, _expected)
    self.assertEqual(_called_with, [foo])