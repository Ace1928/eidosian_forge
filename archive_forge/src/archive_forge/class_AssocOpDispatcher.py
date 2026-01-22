from __future__ import annotations
from operator import attrgetter
from collections import defaultdict
from sympy.utilities.exceptions import sympy_deprecation_warning
from .sympify import _sympify as _sympify_, sympify
from .basic import Basic
from .cache import cacheit
from .sorting import ordered
from .logic import fuzzy_and
from .parameters import global_parameters
from sympy.utilities.iterables import sift
from sympy.multipledispatch.dispatcher import (Dispatcher,
class AssocOpDispatcher:
    """
    Handler dispatcher for associative operators

    .. notes::
       This approach is experimental, and can be replaced or deleted in the future.
       See https://github.com/sympy/sympy/pull/19463.

    Explanation
    ===========

    If arguments of different types are passed, the classes which handle the operation for each type
    are collected. Then, a class which performs the operation is selected by recursive binary dispatching.
    Dispatching relation can be registered by ``register_handlerclass`` method.

    Priority registration is unordered. You cannot make ``A*B`` and ``B*A`` refer to
    different handler classes. All logic dealing with the order of arguments must be implemented
    in the handler class.

    Examples
    ========

    >>> from sympy import Add, Expr, Symbol
    >>> from sympy.core.add import add

    >>> class NewExpr(Expr):
    ...     @property
    ...     def _add_handler(self):
    ...         return NewAdd
    >>> class NewAdd(NewExpr, Add):
    ...     pass
    >>> add.register_handlerclass((Add, NewAdd), NewAdd)

    >>> a, b = Symbol('a'), NewExpr()
    >>> add(a, b) == NewAdd(a, b)
    True

    """

    def __init__(self, name, doc=None):
        self.name = name
        self.doc = doc
        self.handlerattr = '_%s_handler' % name
        self._handlergetter = attrgetter(self.handlerattr)
        self._dispatcher = Dispatcher(name)

    def __repr__(self):
        return '<dispatched %s>' % self.name

    def register_handlerclass(self, classes, typ, on_ambiguity=ambiguity_register_error_ignore_dup):
        """
        Register the handler class for two classes, in both straight and reversed order.

        Paramteters
        ===========

        classes : tuple of two types
            Classes who are compared with each other.

        typ:
            Class which is registered to represent *cls1* and *cls2*.
            Handler method of *self* must be implemented in this class.
        """
        if not len(classes) == 2:
            raise RuntimeError('Only binary dispatch is supported, but got %s types: <%s>.' % (len(classes), str_signature(classes)))
        if len(set(classes)) == 1:
            raise RuntimeError('Duplicate types <%s> cannot be dispatched.' % str_signature(classes))
        self._dispatcher.add(tuple(classes), typ, on_ambiguity=on_ambiguity)
        self._dispatcher.add(tuple(reversed(classes)), typ, on_ambiguity=on_ambiguity)

    @cacheit
    def __call__(self, *args, _sympify=True, **kwargs):
        """
        Parameters
        ==========

        *args :
            Arguments which are operated
        """
        if _sympify:
            args = tuple(map(_sympify_, args))
        handlers = frozenset(map(self._handlergetter, args))
        return self.dispatch(handlers)(*args, _sympify=False, **kwargs)

    @cacheit
    def dispatch(self, handlers):
        """
        Select the handler class, and return its handler method.
        """
        if len(handlers) == 1:
            h, = handlers
            if not isinstance(h, type):
                raise RuntimeError('Handler {!r} is not a type.'.format(h))
            return h
        for i, typ in enumerate(handlers):
            if not isinstance(typ, type):
                raise RuntimeError('Handler {!r} is not a type.'.format(typ))
            if i == 0:
                handler = typ
            else:
                prev_handler = handler
                handler = self._dispatcher.dispatch(prev_handler, typ)
                if not isinstance(handler, type):
                    raise RuntimeError('Dispatcher for {!r} and {!r} must return a type, but got {!r}'.format(prev_handler, typ, handler))
        return handler

    @property
    def __doc__(self):
        docs = ['Multiply dispatched associative operator: %s' % self.name, 'Note that support for this is experimental, see the docs for :class:`AssocOpDispatcher` for details']
        if self.doc:
            docs.append(self.doc)
        s = 'Registered handler classes\n'
        s += '=' * len(s)
        docs.append(s)
        amb_sigs = []
        typ_sigs = defaultdict(list)
        for sigs in self._dispatcher.ordering[::-1]:
            key = self._dispatcher.funcs[sigs]
            typ_sigs[key].append(sigs)
        for typ, sigs in typ_sigs.items():
            sigs_str = ', '.join(('<%s>' % str_signature(sig) for sig in sigs))
            if isinstance(typ, RaiseNotImplementedError):
                amb_sigs.append(sigs_str)
                continue
            s = 'Inputs: %s\n' % sigs_str
            s += '-' * len(s) + '\n'
            s += typ.__name__
            docs.append(s)
        if amb_sigs:
            s = 'Ambiguous handler classes\n'
            s += '=' * len(s)
            docs.append(s)
            s = '\n'.join(amb_sigs)
            docs.append(s)
        return '\n\n'.join(docs)