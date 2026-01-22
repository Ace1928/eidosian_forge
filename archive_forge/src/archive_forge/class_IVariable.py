from pyomo.common.modeling import NoArgumentGiven
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import NumericValue, is_numeric_data, value
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.kernel.set_types import RealSet, IntegerSet
class IVariable(ICategorizedObject, NumericValue):
    """The interface for decision variables"""
    __slots__ = ()
    _valid_domain_types = (RealSet, IntegerSet)
    domain_type = _abstract_readwrite_property(doc='The domain type of the variable (:class:`RealSet` or :class:`IntegerSet`)')
    lb = _abstract_readwrite_property(doc='The lower bound of the variable')
    ub = _abstract_readwrite_property(doc='The upper bound of the variable')
    value = _abstract_readwrite_property(doc='The value of the variable')
    fixed = _abstract_readwrite_property(doc='The fixed status of the variable')
    stale = _abstract_readwrite_property(doc='The stale status of the variable')

    @property
    def bounds(self):
        """Get/Set the bounds as a tuple (lb, ub)."""
        return (self.lb, self.ub)

    @bounds.setter
    def bounds(self, bounds_tuple):
        self.lower, self.upper = bounds_tuple

    @property
    def lb(self):
        """Return the numeric value of the variable lower bound."""
        lb = value(self.lower)
        if lb == _neg_inf:
            return None
        return lb

    @lb.setter
    def lb(self, val):
        self.lower = val

    @property
    def ub(self):
        """Return the numeric value of the variable upper bound."""
        ub = value(self.upper)
        if ub == _pos_inf:
            return None
        return ub

    @ub.setter
    def ub(self, val):
        self.upper = val

    def fix(self, value=NoArgumentGiven):
        """
        Fix the variable. Sets the fixed indicator to
        :const:`True`. An optional value argument will
        update the variable's value before fixing.
        """
        if value is not NoArgumentGiven:
            self.value = value
        self.fixed = True

    def unfix(self):
        """Free the variable. Sets the fixed indicator to
        :const:`False`."""
        self.fixed = False
    free = unfix

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        lb = self.lb
        return lb is not None and lb != _neg_inf

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        ub = self.ub
        return ub is not None and ub != _pos_inf

    @property
    def lslack(self):
        """Lower slack (value - lb). Returns :const:`None` if
        the variable value is :const:`None`."""
        val = self.value
        if val is None:
            return None
        lb = self.lb
        if lb is None:
            lb = _neg_inf
        return val - lb

    @property
    def uslack(self):
        """Upper slack (ub - value). Returns :const:`None` if
        the variable value is :const:`None`."""
        val = self.value
        if val is None:
            return None
        ub = self.ub
        if ub is None:
            ub = _pos_inf
        return ub - val

    @property
    def slack(self):
        """min(lslack, uslack). Returns :const:`None` if
        the variable value is :const:`None`."""
        val = self.value
        if val is None:
            return None
        return min(self.lslack, self.uslack)

    def is_continuous(self):
        """Returns :const:`True` when the domain type is
        :class:`RealSet`."""
        return self.domain_type.get_interval()[2] == 0

    def is_discrete(self):
        """Returns :const:`True` when the domain type is
        :class:`IntegerSet`."""
        return self.domain_type.get_interval()[2] not in (0, None)

    def is_integer(self):
        """Returns :const:`True` when the domain type is
        :class:`IntegerSet`."""
        return self.domain_type.get_interval()[2] == 1

    def is_binary(self):
        """Returns :const:`True` when the domain type is
        :class:`IntegerSet` and the bounds are within
        [0,1]."""
        return self.domain_type.get_interval()[2] == 1 and (self.lb, self.ub) in {(0, 1), (0, 0), (1, 1)}

    def is_fixed(self):
        """Returns :const:`True` if this variable is fixed,
        otherwise returns :const:`False`."""
        return self.fixed

    def is_constant(self):
        """Returns :const:`False` because this is not a
        constant in an expression."""
        return False

    def is_parameter_type(self):
        """Returns :const:`False` because this is not a
        parameter object."""
        return False

    def is_variable_type(self):
        """Returns :const:`True` because this is a
        variable object."""
        return True

    def is_potentially_variable(self):
        """Returns :const:`True` because this is a
        variable."""
        return True

    def polynomial_degree(self):
        """Return the polynomial degree of this
        expression"""
        if self.fixed:
            return 0
        return 1

    def __call__(self, exception=True):
        """Return the value of this variable."""
        if exception and self.value is None:
            raise ValueError('value is None')
        return self.value