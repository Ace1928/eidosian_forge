from .core import unify, reify  # type: ignore[attr-defined]
from .dispatch import dispatch
def unify_object(u, v, s):
    """ Unify two Python objects
    Unifies their type and ``__dict__`` attributes
    >>> # xdoctest: +SKIP
    >>> class Foo(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    ...     def __str__(self):
    ...         return "Foo(%s, %s)"%(str(self.a), str(self.b))
    >>> x = var('x')
    >>> f = Foo(1, x)
    >>> g = Foo(1, 2)
    >>> unify_object(f, g, {})
    {~x: 2}
    """
    if type(u) != type(v):
        return False
    if hasattr(u, '__slots__'):
        return unify([getattr(u, slot) for slot in u.__slots__], [getattr(v, slot) for slot in v.__slots__], s)
    else:
        return unify(u.__dict__, v.__dict__, s)