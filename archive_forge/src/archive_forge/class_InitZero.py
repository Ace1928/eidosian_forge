from pyomo.core.base.var import Var
from pyomo.core.base.transformation import TransformationFactory
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
@TransformationFactory.register('contrib.init_vars_zero', doc='Initialize non-fixed variables to zero.')
class InitZero(IsomorphicTransformation):
    """Initialize non-fixed variables to zero.

    - If setting the variable value to zero will violate a bound, set the
      variable value to the relevant bound value.

    """

    def _apply_to(self, instance, overwrite=False):
        """Apply the transformation.

        Kwargs:
            overwrite: if False, transformation will not overwrite existing
                variable values.
        """
        for var in instance.component_data_objects(ctype=Var, descend_into=True):
            if var.fixed:
                continue
            if var.value is not None and (not overwrite):
                continue
            if var.lb is not None and value(var.lb) > 0:
                var.set_value(value(var.lb))
            elif var.ub is not None and value(var.ub) < 0:
                var.set_value(value(var.ub))
            else:
                var.set_value(0)