from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
class MassAction(RateExpr, UnaryWrapper):
    """Rate-expression of mass-action type

    Notes
    -----
    :meth:`__call__` requires a :class:`Reaction` instance to be passed as ``reaction``
    keyword argument.

    Examples
    --------
    >>> ma = MassAction([3.14])
    >>> from chempy import Reaction
    >>> r = Reaction.from_string('3 A -> B', param=ma)
    >>> r.rate({'A': 2}) == {'A': -75.36, 'B': 25.12}
    True

    """

    def _str(self, *args, **kwargs):
        arg, = self.args
        if isinstance(arg, Symbol):
            uk, = arg.unique_keys
            return "'%s'" % uk
        else:
            return super(MassAction, self)._str(*args, **kwargs)

    def __repr__(self):
        return super(MassAction, self)._str(repr)

    def get_named_keys(self):
        arg, = self.args
        if isinstance(arg, Symbol):
            return arg.args
        else:
            return self.unique_keys
    argument_names = ('rate_constant',)

    def args_dimensionality(self, reaction):
        order = reaction.order()
        return ({'time': -1, 'amount': 1 - order, 'length': 3 * (order - 1)},)

    def active_conc_prod(self, variables, backend=math, reaction=None):
        result = 1
        for k, v in reaction.reac.items():
            result *= variables[k] ** v
        return result

    def rate_coeff(self, variables, backend=math, **kwargs):
        rat_c, = self.all_args(variables, backend=backend, **kwargs)
        return rat_c

    def __call__(self, variables, backend=math, reaction=None, **kwargs):
        return self.rate_coeff(variables, backend=backend, reaction=reaction) * self.active_conc_prod(variables, backend=backend, reaction=reaction, **kwargs)

    def string(self, *args, **kwargs):
        if self.args is None and len(self.unique_keys) == 1:
            return self.unique_keys[0]
        else:
            return super(MassAction, self).string(*args, **kwargs)

    @classmethod
    @deprecated(use_instead='MassAction.from_callback')
    def subclass_from_callback(cls, cb, cls_attrs=None):
        """Override MassAction.__call__"""
        _RateExpr = super(MassAction, cls).subclass_from_callback(cb, cls_attrs=cls_attrs)

        def wrapper(*args, **kwargs):
            obj = _RateExpr(*args, **kwargs)
            return cls(obj)
        return wrapper

    @classmethod
    def from_callback(cls, callback, attr='__call__', **kwargs):
        Wrapper = RateExpr.from_callback(callback, attr=attr, **kwargs)
        return lambda *args, **kwargs: MassAction(Wrapper(*args, **kwargs))