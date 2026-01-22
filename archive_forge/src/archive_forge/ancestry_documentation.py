from __future__ import annotations
from typing import TYPE_CHECKING, cast
from more_itertools import unique_everseen

    Generator over all subclasses of a given class, in depth-first order.

    >>> bool in list(iter_subclasses(int))
    True
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>>
    >>> for cls in iter_subclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
    >>> # get ALL classes currently defined
    >>> res = [cls.__name__ for cls in iter_subclasses(object)]
    >>> 'type' in res
    True
    >>> 'tuple' in res
    True
    >>> len(res) > 100
    True
    