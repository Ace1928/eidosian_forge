from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
class ConditionalNeqSys(_NeqSysBase):
    """ Collect multiple systems of non-linear equations with different
    conditionals.

    If a problem in a fixed number of variables is described by different
    systems of equations for different domains, then this class may be used
    to describe that set of systems.

    The user provides a set of conditions which governs what system of
    equations to apply. The set of conditions then represent a vector
    of booleans which is passed to a user provided factory function of
    NeqSys instances. The conditions may be asymmetrical (each condition
    consits of two callbacks, one for evaluating when the condition was
    previously ``False``, and one when it was previously ``True``. The motivation
    for this asymmetry is that a user may want to introduce a tolerance for
    numerical noise in the solution (and avoid possibly endless loops).

    If ``fastcache`` is available an LRU cache will be used for
    ``neqsys_factory``, it is therefore important that the factory function
    is idempotent.

    Parameters
    ----------
    condition_cb_pairs : list of (callback, callback) tuples
        Callbacks should have the signature: ``f(x, p) -> bool``.
    neqsys_factory : callback
        Should have the signature ``f(conds) -> NeqSys instance``,
        where conds is a list of bools.
    names : list of strings

    Examples
    --------
    >>> from math import sin, pi
    >>> f_a = lambda x, p: [sin(p[0]*x[0])]  # when x <= 0
    >>> f_b = lambda x, p: [x[0]*(p[1]-x[0])]  # when x >= 0
    >>> factory = lambda conds: NeqSys(1, 1, f_b) if conds[0] else NeqSys(
    ...     1, 1, f_a)
    >>> cneqsys = ConditionalNeqSys([(lambda x, p: x[0] > 0,  # no 0-switch
    ...                               lambda x, p: x[0] >= 0)],  # no 0-switch
    ...                             factory)
    >>> x, sol = cneqsys.solve([0], [pi, 3])
    >>> assert sol['success']
    >>> print(x)  # doctest: +NORMALIZE_WHITESPACE
    [ 0.]
    >>> x, sol = cneqsys.solve([-1.4], [pi, 3])
    >>> assert sol['success']
    >>> print(x)  # doctest: +NORMALIZE_WHITESPACE
    [-1.]
    >>> x, sol = cneqsys.solve([2], [pi, 3])
    >>> assert sol['success']
    >>> print(x)  # doctest: +NORMALIZE_WHITESPACE
    [ 3.]
    >>> x, sol = cneqsys.solve([7], [pi, 3])
    >>> assert sol['success']
    >>> print(x)  # doctest: +NORMALIZE_WHITESPACE
    [ 3.]

    """

    def __init__(self, condition_cb_pairs, neqsys_factory, **kwargs):
        super(ConditionalNeqSys, self).__init__(**kwargs)
        self.condition_cb_pairs = condition_cb_pairs
        self.neqsys_factory = _cache_it(neqsys_factory)

    def get_conds(self, x, params, prev_conds=None):
        if prev_conds is None:
            prev_conds = [False] * len(self.condition_cb_pairs)
        return tuple([bw(x, params) if prev else fw(x, params) for prev, (fw, bw) in zip(prev_conds, self.condition_cb_pairs)])

    def solve(self, x0, params=(), internal_x0=None, solver=None, conditional_maxiter=20, initial_conditions=None, **kwargs):
        """ Solve the problem (systems of equations)

        Parameters
        ----------
        x0 : array
            Guess.
        params : array
            See :meth:`NeqSys.solve`.
        internal_x0 : array
            See :meth:`NeqSys.solve`.
        solver : str or callable or iterable of such.
            See :meth:`NeqSys.solve`.
        conditional_maxiter : int
            Maximum number of switches between conditions.
        initial_conditions : iterable of bools
            Corresponding conditions to ``x0``
        \\*\\*kwargs :
            Keyword arguments passed on to :meth:`NeqSys.solve`.

        """
        if initial_conditions is not None:
            conds = initial_conditions
        else:
            conds = self.get_conds(x0, params, initial_conditions)
        idx, nfev, njev = (0, 0, 0)
        while idx < conditional_maxiter:
            neqsys = self.neqsys_factory(conds)
            x0, info = neqsys.solve(x0, params, internal_x0, solver, **kwargs)
            if idx == 0:
                internal_x0 = None
            nfev += info['nfev']
            njev += info.get('njev', 0)
            new_conds = self.get_conds(x0, params, conds)
            if new_conds == conds:
                break
            else:
                conds = new_conds
            idx += 1
        if idx == conditional_maxiter:
            raise Exception('Solving failed, conditional_maxiter reached')
        self.internal_x = info['x']
        self.internal_params = neqsys.internal_params
        result = {'x': info['x'], 'success': info['success'], 'conditions': conds, 'nfev': nfev, 'njev': njev}
        if 'fun' in info:
            result['fun'] = info['fun']
        return (x0, result)

    def post_process(self, x, params, conds=None):
        if conds is None:
            conds = self.get_conds(x, params)
        return self.neqsys_factory(conds).post_process(x, params)
    post_process.__doc__ = NeqSys.post_process.__doc__

    def pre_process(self, x, params, conds=None):
        if conds is None:
            conds = self.get_conds(x, params)
        return self.neqsys_factory(conds).pre_process(x, params)
    pre_process.__doc__ = NeqSys.pre_process.__doc__

    def f_cb(self, x, params, conds=None):
        if conds is None:
            conds = self.get_conds(x, params)
        return self.neqsys_factory(conds).f_cb(x, params)