from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.transformation import TransformationFactory
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.config import (
from pyomo.common.errors import InfeasibleConstraintException
@TransformationFactory.register('contrib.propagate_eq_var_bounds', doc='Propagate variable bounds for equalities of type x = y.')
@document_kwargs_from_configdict('CONFIG')
class VarBoundPropagator(IsomorphicTransformation):
    """Propagate variable bounds for equalities of type :math:`x = y`.

    If :math:`x` has a tighter bound then :math:`y`, then this transformation
    will adjust the bounds on :math:`y` to match those of :math:`x`.

    Keyword arguments below are specified for the ``apply_to`` and
    ``create_using`` functions.

    """
    CONFIG = ConfigBlock()
    CONFIG.declare('tmp', ConfigValue(default=False, domain=bool, description='True to store the set of transformed variables and their old states so that they can be later restored.'))

    def _apply_to(self, instance, **kwds):
        config = self.CONFIG(kwds)
        if config.tmp and (not hasattr(instance, '_tmp_propagate_original_bounds')):
            instance._tmp_propagate_original_bounds = Suffix(direction=Suffix.LOCAL)
        eq_var_map, relevant_vars = _build_equality_set(instance)
        processed = ComponentSet()
        for var in relevant_vars:
            if var in processed:
                continue
            var_equality_set = eq_var_map.get(var, ComponentSet([var]))
            lbs = [v.lb for v in var_equality_set if v.has_lb()]
            max_lb = max(lbs) if len(lbs) > 0 else None
            ubs = [v.ub for v in var_equality_set if v.has_ub()]
            min_ub = min(ubs) if len(ubs) > 0 else None
            if max_lb is not None and min_ub is not None and (max_lb > min_ub):
                v1 = next((v for v in var_equality_set if v.lb == max_lb))
                v2 = next((v for v in var_equality_set if v.ub == min_ub))
                raise InfeasibleConstraintException('Variable {} has a lower bound {} > the upper bound {} of variable {}, but they are linked by equality constraints.'.format(v1.name, value(v1.lb), value(v2.ub), v2.name))
            for v in var_equality_set:
                if config.tmp:
                    instance._tmp_propagate_original_bounds[v] = (v.lb, v.ub)
                v.setlb(max_lb)
                v.setub(min_ub)
            processed.update(var_equality_set)

    def revert(self, instance):
        """Revert variable bounds."""
        for v in instance._tmp_propagate_original_bounds:
            old_LB, old_UB = instance._tmp_propagate_original_bounds[v]
            v.setlb(old_LB)
            v.setub(old_UB)
        del instance._tmp_propagate_original_bounds