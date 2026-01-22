from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.core.expr.logical_expr import special_boolean_atom_types
from pyomo.core.expr.numvalue import native_types, value
from pyomo.core.expr.sympy_tools import (
def _convert_children_to_literals(special_atom, bool_varlist, bool_var_to_special_atoms):
    """If the child logical constraints are not literals, substitute
    augmented boolean variables.

    Same return types as to_cnf() function.

    """
    new_args = [special_atom.args[0]]
    new_statements = []
    need_new_expression = False
    for child in special_atom.args[1:]:
        if type(child) in native_types or not child.is_expression_type():
            new_args.append(child)
        else:
            need_new_expression = True
            new_indicator = bool_varlist.add()
            if type(child) in special_boolean_atom_types:
                child_cnf = _convert_children_to_literals(child, bool_varlist, bool_var_to_special_atoms)
                bool_var_to_special_atoms[new_indicator] = child_cnf[0]
            else:
                child_cnf = to_cnf(new_indicator.equivalent_to(child), bool_varlist, bool_var_to_special_atoms)
                new_statements.append(child_cnf[0])
            new_args.append(new_indicator)
            new_statements.extend(child_cnf[1:])
    if need_new_expression:
        new_atom_with_literals = special_atom.__class__(new_args)
        return [new_atom_with_literals] + new_statements
    else:
        return [special_atom]