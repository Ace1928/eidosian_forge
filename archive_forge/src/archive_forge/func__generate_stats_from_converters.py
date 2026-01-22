from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
def _generate_stats_from_converters(converters, node):
    stats = []
    for converter in converters:
        tree = converter.generate_cy_utility_code()
        ufunc_node = get_cfunc_from_tree(tree)
        converter.global_scope.utility_code_list.extend(tree.scope.utility_code_list)
        stats.append(ufunc_node)
    stats.append(generate_ufunc_initialization(converters, stats, node))
    return stats