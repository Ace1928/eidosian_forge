import unittest
from zope.interface.tests import OptimizationTestMixin
def test_subscribers_wo_provided(self):
    from zope.interface.declarations import implementer
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))

    @implementer(IFoo)
    class Foo:
        pass
    foo = Foo()
    registry = self._makeRegistry(IFoo, IBar)
    registry = self._makeRegistry(IFoo, IBar)
    subr = self._makeSubregistry()
    _called = {}

    def _factory1(context):
        _called.setdefault('_factory1', []).append(context)

    def _factory2(context):
        _called.setdefault('_factory2', []).append(context)
    subr._subscribers = [{}, {IFoo: {None: {'': (_factory1, _factory2)}}}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    alb.subscriptions = alb._uncached_subscriptions
    subr._v_lookup = alb
    result = alb.subscribers((foo,), None)
    self.assertEqual(result, ())
    self.assertEqual(_called, {'_factory1': [foo], '_factory2': [foo]})