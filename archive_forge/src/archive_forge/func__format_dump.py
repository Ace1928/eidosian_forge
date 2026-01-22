from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines
def _format_dump(node: NodeOrLeaf, indent: str='', top_level: bool=True) -> str:
    result = ''
    node_type = type(node).__name__
    if isinstance(node, Leaf):
        result += f'{indent}{node_type}('
        if isinstance(node, ErrorLeaf):
            result += f'{node.token_type!r}, '
        elif isinstance(node, TypedLeaf):
            result += f'{node.type!r}, '
        result += f'{node.value!r}, {node.start_pos!r}'
        if node.prefix:
            result += f', prefix={node.prefix!r}'
        result += ')'
    elif isinstance(node, BaseNode):
        result += f'{indent}{node_type}('
        if isinstance(node, Node):
            result += f'{node.type!r}, '
        result += '['
        if newline:
            result += '\n'
        for child in node.children:
            result += _format_dump(child, indent=indent + indent_string, top_level=False)
        result += f'{indent}])'
    else:
        raise TypeError(f'unsupported node encountered: {node!r}')
    if not top_level:
        if newline:
            result += ',\n'
        else:
            result += ', '
    return result