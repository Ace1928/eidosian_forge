from pyomo.core.base.var import Var
from pyomo.core.base.transformation import TransformationFactory
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
Apply the transformation.

        Kwargs:
            overwrite: if False, transformation will not overwrite existing
                variable values.
        