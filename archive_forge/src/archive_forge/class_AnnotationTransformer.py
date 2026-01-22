from __future__ import annotations
import ast
import builtins
import sys
import typing
from ast import (
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, overload
class AnnotationTransformer(NodeTransformer):
    type_substitutions: ClassVar[dict[str, tuple[str, str]]] = {'builtins.dict': ('typing', 'Dict'), 'builtins.list': ('typing', 'List'), 'builtins.tuple': ('typing', 'Tuple'), 'builtins.set': ('typing', 'Set'), 'builtins.frozenset': ('typing', 'FrozenSet')}

    def __init__(self, transformer: TypeguardTransformer):
        self.transformer = transformer
        self._memo = transformer._memo
        self._level = 0

    def visit(self, node: AST) -> Any:
        if isinstance(node, expr) and self._memo.name_matches(node, *literal_names):
            return node
        self._level += 1
        new_node = super().visit(node)
        self._level -= 1
        if isinstance(new_node, Expression) and (not hasattr(new_node, 'body')):
            return None
        if self._level == 0 and isinstance(new_node, expr) and self._memo.name_matches(new_node, *anytype_names):
            return None
        return new_node

    def visit_BinOp(self, node: BinOp) -> Any:
        self.generic_visit(node)
        if isinstance(node.op, BitOr):
            if not hasattr(node, 'left') or not hasattr(node, 'right'):
                return None
            if self._memo.name_matches(node.left, *anytype_names):
                return node.left
            elif self._memo.name_matches(node.right, *anytype_names):
                return node.right
            if sys.version_info < (3, 10):
                union_name = self.transformer._get_import('typing', 'Union')
                return Subscript(value=union_name, slice=Index(Tuple(elts=[node.left, node.right], ctx=Load()), ctx=Load()), ctx=Load())
        return node

    def visit_Attribute(self, node: Attribute) -> Any:
        if self._memo.is_ignored_name(node):
            return None
        return node

    def visit_Subscript(self, node: Subscript) -> Any:
        if self._memo.is_ignored_name(node.value):
            return None
        if node.slice:
            if isinstance(node.slice, Index):
                slice_value = node.slice.value
            else:
                slice_value = node.slice
            if isinstance(slice_value, Tuple):
                if self._memo.name_matches(node.value, *annotated_names):
                    items = cast(typing.List[expr], [self.visit(slice_value.elts[0])] + slice_value.elts[1:])
                else:
                    items = cast(typing.List[expr], [self.visit(item) for item in slice_value.elts])
                if self._memo.name_matches(node.value, 'typing.Union') and any((item is None or (isinstance(item, expr) and self._memo.name_matches(item, *anytype_names)) for item in items)):
                    return None
                if all((item is None for item in items)):
                    return node.value
                for index, item in enumerate(items):
                    if item is None:
                        items[index] = self.transformer._get_import('typing', 'Any')
                slice_value.elts = items
            else:
                self.generic_visit(node)
                if self._memo.name_matches(node.value, 'typing.Optional') and (not hasattr(node, 'slice')):
                    return None
                if sys.version_info >= (3, 9) and (not hasattr(node, 'slice')):
                    return node.value
                elif sys.version_info < (3, 9) and (not hasattr(node.slice, 'value')):
                    return node.value
        return node

    def visit_Name(self, node: Name) -> Any:
        if self._memo.is_ignored_name(node):
            return None
        if sys.version_info < (3, 9):
            for typename, substitute in self.type_substitutions.items():
                if self._memo.name_matches(node, typename):
                    new_node = self.transformer._get_import(*substitute)
                    return copy_location(new_node, node)
        return node

    def visit_Call(self, node: Call) -> Any:
        return node

    def visit_Constant(self, node: Constant) -> Any:
        if isinstance(node.value, str):
            expression = ast.parse(node.value, mode='eval')
            new_node = self.visit(expression)
            if new_node:
                return copy_location(new_node.body, node)
            else:
                return None
        return node