import re
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
def _create_input_dict(function_graph, func_arg_placeholders, initial_value=None):
    """Create a mapping from graph tensor names to function tensor names."""
    if initial_value is None:
        input_dict = {}
    else:
        input_dict = dict(initial_value)
    for op in function_graph.get_operations():
        if _is_in_placeholders(op, func_arg_placeholders):
            input_dict[op.name] = op.name
        else:
            op_def = _get_op_def(op)
            attrs = _get_node_def(op).attr
            o = 0
            for arg_def in op_def.output_arg:
                if arg_def.number_attr:
                    num = attrs[arg_def.number_attr].i
                elif arg_def.type_list_attr:
                    num = len(attrs[arg_def.type_list_attr].list.type)
                else:
                    num = 1
                for i in range(num):
                    result = '%s:%s:%d' % (op.name, arg_def.name, i)
                    input_dict[op.values()[o].name] = result
                    if o == 0:
                        input_dict[op.name] = result
                    o += 1
    return input_dict