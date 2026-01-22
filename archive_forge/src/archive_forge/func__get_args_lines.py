from ray.dag import DAGNode
from ray.util.annotations import DeveloperAPI
def _get_args_lines(bound_args):
    """Pretty prints bounded args of a DAGNode, and recursively handle
    DAGNode in list / dict containers.
    """
    indent = _get_indentation()
    lines = []
    for arg in bound_args:
        if isinstance(arg, DAGNode):
            node_repr_lines = str(arg).split('\n')
            for node_repr_line in node_repr_lines:
                lines.append(f'{indent}' + node_repr_line)
        elif isinstance(arg, list):
            for ele in arg:
                node_repr_lines = str(ele).split('\n')
                for node_repr_line in node_repr_lines:
                    lines.append(f'{indent}' + node_repr_line)
        elif isinstance(arg, dict):
            for _, val in arg.items():
                node_repr_lines = str(val).split('\n')
                for node_repr_line in node_repr_lines:
                    lines.append(f'{indent}' + node_repr_line)
        else:
            lines.append(f'{indent}' + str(arg) + ', ')
    if len(lines) == 0:
        args_line = '[]'
    else:
        args_line = '['
        for args in lines:
            args_line += f'\n{indent}{args}'
        args_line += f'\n{indent}]'
    return args_line