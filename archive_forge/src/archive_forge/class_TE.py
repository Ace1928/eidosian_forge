import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
class TE(object):
    """te(s1, .., sn, constraints=None)

    Generates smooth of several covariates as a tensor product of the bases
    of marginal univariate smooths ``s1, .., sn``. The marginal smooths are
    required to transform input univariate data into some kind of smooth
    functions basis producing a 2-d array output with the ``(i, j)`` element
    corresponding to the value of the ``j`` th basis function at the ``i`` th
    data point.
    The resulting basis dimension is the product of the basis dimensions of
    the marginal smooths. The usual usage is something like::

      y ~ 1 + te(cr(x1, df=5), cc(x2, df=6), constraints='center')

    to fit ``y`` as a smooth function of both ``x1`` and ``x2``, with a natural
    cubic spline for ``x1`` marginal smooth and a cyclic cubic spline for
    ``x2`` (and centering constraint absorbed in the resulting design matrix).

    :arg constraints: Either a 2-d array defining general linear constraints
     (that is ``np.dot(constraints, betas)`` is zero, where ``betas`` denotes
     the array of *initial* parameters, corresponding to the *initial*
     unconstrained design matrix), or the string
     ``'center'`` indicating that we should apply a centering constraint
     (this constraint will be computed from the input data, remembered and
     re-used for prediction from the fitted model).
     The constraints are absorbed in the resulting design matrix which means
     that the model is actually rewritten in terms of
     *unconstrained* parameters. For more details see :ref:`spline-regression`.

    Using this function requires scipy be installed.

    .. note:: This function reproduce the tensor product smooth 'te' as
      implemented in the R package 'mgcv' (GAM modelling).
      See also 'Generalized Additive Models', Simon N. Wood, 2006, pp 158-163

    .. versionadded:: 0.3.0
    """

    def __init__(self):
        self._tmp = {}
        self._constraints = None

    def memorize_chunk(self, *args, **kwargs):
        constraints = self._tmp.setdefault('constraints', kwargs.get('constraints'))
        if safe_string_eq(constraints, 'center'):
            args_2d = []
            for arg in args:
                arg = atleast_2d_column_default(arg)
                if arg.ndim != 2:
                    raise ValueError('Each tensor product argument must be a 2-d array or 1-d vector.')
                args_2d.append(arg)
            tp = _row_tensor_product(args_2d)
            self._tmp.setdefault('count', 0)
            self._tmp['count'] += tp.shape[0]
            chunk_sum = np.atleast_2d(tp.sum(axis=0))
            self._tmp.setdefault('sum', np.zeros(chunk_sum.shape))
            self._tmp['sum'] += chunk_sum

    def memorize_finish(self):
        tmp = self._tmp
        constraints = self._tmp['constraints']
        del self._tmp
        if constraints is not None:
            if safe_string_eq(constraints, 'center'):
                constraints = np.atleast_2d(tmp['sum'] / tmp['count'])
            else:
                constraints = np.atleast_2d(constraints)
                if constraints.ndim != 2:
                    raise ValueError('Constraints must be 2-d array or 1-d vector.')
        self._constraints = constraints

    def transform(self, *args, **kwargs):
        args_2d = []
        for arg in args:
            arg = atleast_2d_column_default(arg)
            if arg.ndim != 2:
                raise ValueError('Each tensor product argument must be a 2-d array or 1-d vector.')
            args_2d.append(arg)
        return _get_te_dmatrix(args_2d, self._constraints)
    __getstate__ = no_pickling