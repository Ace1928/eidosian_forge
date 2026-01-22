import fractions
import numpy as np
import pytest
import sympy
import cirq
def _assert_consistent_resolution(v, resolved):
    """Asserts that parameter resolution works consistently.

    The ParamResolver.value_of method can resolve any Sympy expression -
    subclasses of sympy.Basic. In the generic case, it calls `sympy.Basic.subs`
    to substitute symbols with values specified in a dict, which is known to be
    very slow. Instead value_of defines a pass-through shortcut for known
    numeric types. For a given value `v` it is asserted that value_of resolves
    it to `resolved`, with the exact type of `resolved`.`subs_called` indicates
    whether it is expected to have `subs` called or not during the resolution.

    Args:
        v: the value to resolve
        resolved: the expected resolution result

    Raises:
        AssertionError in case resolution assertion fail.
    """

    class SubsAwareSymbol(sympy.Symbol):
        """A Symbol that registers a call to its `subs` method."""

        def __init__(self, sym: str):
            self.called = False
            self.symbol = sympy.Symbol(sym)

        def subs(self, *args, **kwargs):
            self.called = True
            return self.symbol.subs(*args, **kwargs)
    r = cirq.ParamResolver({'a': v})
    s = SubsAwareSymbol('a')
    assert r.value_of(s) == resolved, f'expected {resolved}, got {r.value_of(s)}'
    assert not s.called, f"For pass-through type {type(v)} sympy.subs shouldn't have been called."
    assert isinstance(r.value_of(s), type(resolved)), f'expected {type(resolved)} got {type(r.value_of(s))}'
    assert r.value_of('a') == resolved, f'expected {resolved}, got {r.value_of('a')}'
    assert isinstance(r.value_of('a'), type(resolved)), f'expected {type(resolved)} got {type(r.value_of('a'))}'
    assert r.value_of(v) == resolved, f'expected {resolved}, got {r.value_of(v)}'
    assert isinstance(r.value_of(v), type(resolved)), f'expected {type(resolved)} got {type(r.value_of(v))}'