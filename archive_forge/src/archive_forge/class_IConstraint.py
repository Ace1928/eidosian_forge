from pyomo.core.expr.numvalue import (
from pyomo.core.expr.expr_common import ExpressionType
from pyomo.core.expr.relational_expr import (
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readonly_property
from pyomo.core.kernel.container_utils import define_simple_containers
class IConstraint(ICategorizedObject):
    """The interface for constraints"""
    __slots__ = ()
    body = _abstract_readonly_property(doc='The expression for the body of the constraint')
    lower = _abstract_readonly_property(doc='The expression for the lower bound of the constraint')
    upper = _abstract_readonly_property(doc='The expression for the upper bound of the constraint')
    lb = _abstract_readonly_property(doc='The value of the lower bound of the constraint')
    ub = _abstract_readonly_property(doc='The value of the upper bound of the constraint')
    rhs = _abstract_readonly_property(doc='The right-hand side of the constraint')
    equality = _abstract_readonly_property(doc='A boolean indicating whether this is an equality constraint')
    _linear_canonical_form = _abstract_readonly_property(doc='Indicates whether or not the class or instance provides the properties that define the linear canonical form of a constraint')

    def __call__(self, exception=True):
        """Compute the value of the body of this constraint."""
        if exception and self.body is None:
            raise ValueError('constraint body is None')
        elif self.body is None:
            return None
        return self.body(exception=exception)

    @property
    def lslack(self):
        """Lower slack (body - lb). Returns :const:`None` if
        a value for the body can not be computed."""
        body = self(exception=False)
        if body is None:
            return None
        lb = self.lb
        if lb is None:
            lb = _neg_inf
        else:
            lb = value(lb)
        return body - lb

    @property
    def uslack(self):
        """Upper slack (ub - body). Returns :const:`None` if
        a value for the body can not be computed."""
        body = self(exception=False)
        if body is None:
            return None
        ub = self.ub
        if ub is None:
            ub = _pos_inf
        else:
            ub = value(ub)
        return ub - body

    @property
    def slack(self):
        """min(lslack, uslack). Returns :const:`None` if a
        value for the body can not be computed."""
        body = self(exception=False)
        if body is None:
            return None
        return min(self.lslack, self.uslack)

    @property
    def expr(self):
        """Get the expression on this constraint."""
        body_expr = self.body
        if body_expr is None:
            return None
        if self.equality:
            return body_expr == self.rhs
        else:
            if self.lb is None:
                return body_expr <= self.ub
            elif self.ub is None:
                return self.lb <= body_expr
            return RangedExpression((self.lb, body_expr, self.ub), (False, False))

    @property
    def bounds(self):
        """The bounds of the constraint as a tuple (lb, ub)"""
        return (self.lb, self.ub)

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        lb = self.lb
        return lb is not None and value(lb) != float('-inf')

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        ub = self.ub
        return ub is not None and value(ub) != float('inf')