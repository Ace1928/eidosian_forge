from math import fabs
from pyomo.common.collections import ComponentSet
from pyomo.core import TransformationFactory, value, Constraint, Block
def _disjunction_to_str(disjunction):
    pretty = []
    for disjunct in disjunction.disjuncts:
        exprs = []
        for cons in disjunct.component_data_objects(Constraint, active=True, descend_into=Block):
            exprs.append(str(cons.expr))
        pretty.append('[%s]' % ', '.join(exprs))
    return ' v '.join(pretty)