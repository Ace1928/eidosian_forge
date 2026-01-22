import ast
import sys
import warnings
from typing import Iterable, Iterator, List, Set, Tuple
from black.mode import VERSION_TO_FEATURES, Feature, TargetVersion, supports_feature
from black.nodes import syms
from blib2to3 import pygram
from blib2to3.pgen2 import driver
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.parse import ParseError
from blib2to3.pgen2.tokenize import TokenError
from blib2to3.pytree import Leaf, Node
def _stringify_ast(node: ast.AST, parent_stack: List[ast.AST]) -> Iterator[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and (node.kind == 'u'):
        node.kind = None
    yield f'{'    ' * len(parent_stack)}{node.__class__.__name__}('
    for field in sorted(node._fields):
        if isinstance(node, ast.TypeIgnore):
            break
        try:
            value: object = getattr(node, field)
        except AttributeError:
            continue
        yield f'{'    ' * (len(parent_stack) + 1)}{field}='
        if isinstance(value, list):
            for item in value:
                if field == 'targets' and isinstance(node, ast.Delete) and isinstance(item, ast.Tuple):
                    for elt in item.elts:
                        yield from _stringify_ast_with_new_parent(elt, parent_stack, node)
                elif isinstance(item, ast.AST):
                    yield from _stringify_ast_with_new_parent(item, parent_stack, node)
        elif isinstance(value, ast.AST):
            yield from _stringify_ast_with_new_parent(value, parent_stack, node)
        else:
            normalized: object
            if isinstance(node, ast.Constant) and field == 'value' and isinstance(value, str) and (len(parent_stack) >= 2) and isinstance(parent_stack[-1], ast.Expr):
                normalized = _normalize('\n', value)
            elif field == 'type_comment' and isinstance(value, str):
                normalized = value.rstrip()
            else:
                normalized = value
            yield f'{'    ' * (len(parent_stack) + 1)}{normalized!r},  # {value.__class__.__name__}'
    yield f'{'    ' * len(parent_stack)})  # /{node.__class__.__name__}'