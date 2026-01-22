from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
def _visit_binop_node(self, node):
    self._process_children(node)
    special_method_name = find_special_method_for_binary_operator(node.operator)
    if special_method_name:
        operand1, operand2 = (node.operand1, node.operand2)
        if special_method_name == '__contains__':
            operand1, operand2 = (operand2, operand1)
        elif special_method_name == '__div__':
            if Future.division in self.current_env().global_scope().context.future_directives:
                special_method_name = '__truediv__'
        obj_type = operand1.type
        if obj_type.is_builtin_type:
            type_name = obj_type.name
        else:
            type_name = 'object'
        node = self._dispatch_to_method_handler(special_method_name, None, False, type_name, node, None, [operand1, operand2], None)
    return node