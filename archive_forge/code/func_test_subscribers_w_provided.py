import unittest
from zope.interface.tests import OptimizationTestMixin
def test_subscribers_w_provided(self):
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
    _exp1, _exp2 = (object(), object())

    def _factory1(context):
        _called.setdefault('_factory1', []).append(context)
        return _exp1

    def _factory2(context):
        _called.setdefault('_factory2', []).append(context)
        return _exp2

    def _side_effect_only(context):
        _called.setdefault('_side_effect_only', []).append(context)
    subr._subscribers = [{}, {IFoo: {IBar: {'': (_factory1, _factory2, _side_effect_only)}}}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    alb.subscriptions = alb._uncached_subscriptions
    subr._v_lookup = alb
    result = alb.subscribers((foo,), IBar)
    self.assertEqual(result, [_exp1, _exp2])
    self.assertEqual(_called, {'_factory1': [foo], '_factory2': [foo], '_side_effect_only': [foo]})