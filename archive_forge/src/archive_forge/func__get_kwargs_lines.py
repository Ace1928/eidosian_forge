from ray.dag import DAGNode
from ray.util.annotations import DeveloperAPI
def _get_kwargs_lines(bound_kwargs):
    """Pretty prints bounded kwargs of a DAGNode, and recursively handle
    DAGNode in list / dict containers.
    """
    if not bound_kwargs:
        return '{}'
    indent = _get_indentation()
    kwargs_lines = []
    for key, val in bound_kwargs.items():
        if isinstance(val, DAGNode):
            node_repr_lines = str(val).split('\n')
            for index, node_repr_line in enumerate(node_repr_lines):
                if index == 0:
                    kwargs_lines.append(f'{indent}{key}:' + f'{indent}' + node_repr_line)
                else:
                    kwargs_lines.append(f'{indent}{indent}' + node_repr_line)
        elif isinstance(val, list):
            for ele in val:
                node_repr_lines = str(ele).split('\n')
                for node_repr_line in node_repr_lines:
                    kwargs_lines.append(f'{indent}' + node_repr_line)
        elif isinstance(val, dict):
            for _, inner_val in val.items():
                node_repr_lines = str(inner_val).split('\n')
                for node_repr_line in node_repr_lines:
                    kwargs_lines.append(f'{indent}' + node_repr_line)
        else:
            kwargs_lines.append(val)
    if len(kwargs_lines) > 0:
        kwargs_line = '{'
        for line in kwargs_lines:
            kwargs_line += f'\n{indent}{line}'
        kwargs_line += f'\n{indent}}}'
    else:
        kwargs_line = '{}'
    return kwargs_line