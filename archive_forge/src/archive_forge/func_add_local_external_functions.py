from pyomo.core.base.block import Block
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.expression import Expression
from pyomo.core.base.external import ExternalFunction
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
def add_local_external_functions(block):
    ef_exprs = []
    for comp in block.component_data_objects((Constraint, Expression), active=True):
        ef_exprs.extend(identify_external_functions(comp.expr))
    unique_functions = []
    fcn_set = set()
    for expr in ef_exprs:
        fcn = expr._fcn
        data = (fcn._library, fcn._function)
        if data not in fcn_set:
            fcn_set.add(data)
            unique_functions.append(data)
    fcn_comp_map = {}
    for lib, name in unique_functions:
        comp_name = unique_component_name(block, '_' + name)
        comp = ExternalFunction(library=lib, function=name)
        block.add_component(comp_name, comp)
        fcn_comp_map[lib, name] = comp
    return fcn_comp_map