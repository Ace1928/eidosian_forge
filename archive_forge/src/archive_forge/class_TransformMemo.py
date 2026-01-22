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
@dataclass
class TransformMemo:
    node: Module | ClassDef | FunctionDef | AsyncFunctionDef | None
    parent: TransformMemo | None
    path: tuple[str, ...]
    joined_path: Constant = field(init=False)
    return_annotation: expr | None = None
    yield_annotation: expr | None = None
    send_annotation: expr | None = None
    is_async: bool = False
    local_names: set[str] = field(init=False, default_factory=set)
    imported_names: dict[str, str] = field(init=False, default_factory=dict)
    ignored_names: set[str] = field(init=False, default_factory=set)
    load_names: defaultdict[str, dict[str, Name]] = field(init=False, default_factory=lambda: defaultdict(dict))
    has_yield_expressions: bool = field(init=False, default=False)
    has_return_expressions: bool = field(init=False, default=False)
    memo_var_name: Name | None = field(init=False, default=None)
    should_instrument: bool = field(init=False, default=True)
    variable_annotations: dict[str, expr] = field(init=False, default_factory=dict)
    configuration_overrides: dict[str, Any] = field(init=False, default_factory=dict)
    code_inject_index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        elements: list[str] = []
        memo = self
        while isinstance(memo.node, (ClassDef, FunctionDef, AsyncFunctionDef)):
            elements.insert(0, memo.node.name)
            if not memo.parent:
                break
            memo = memo.parent
            if isinstance(memo.node, (FunctionDef, AsyncFunctionDef)):
                elements.insert(0, '<locals>')
        self.joined_path = Constant('.'.join(elements))
        if self.node:
            for index, child in enumerate(self.node.body):
                if isinstance(child, ImportFrom) and child.module == '__future__':
                    continue
                elif isinstance(child, Expr) and isinstance(child.value, Constant) and isinstance(child.value.value, str):
                    continue
                self.code_inject_index = index
                break

    def get_unused_name(self, name: str) -> str:
        memo: TransformMemo | None = self
        while memo is not None:
            if name in memo.local_names:
                memo = self
                name += '_'
            else:
                memo = memo.parent
        self.local_names.add(name)
        return name

    def is_ignored_name(self, expression: expr | Expr | None) -> bool:
        top_expression = expression.value if isinstance(expression, Expr) else expression
        if isinstance(top_expression, Attribute) and isinstance(top_expression.value, Name):
            name = top_expression.value.id
        elif isinstance(top_expression, Name):
            name = top_expression.id
        else:
            return False
        memo: TransformMemo | None = self
        while memo is not None:
            if name in memo.ignored_names:
                return True
            memo = memo.parent
        return False

    def get_memo_name(self) -> Name:
        if not self.memo_var_name:
            self.memo_var_name = Name(id='memo', ctx=Load())
        return self.memo_var_name

    def get_import(self, module: str, name: str) -> Name:
        if module in self.load_names and name in self.load_names[module]:
            return self.load_names[module][name]
        qualified_name = f'{module}.{name}'
        if name in self.imported_names and self.imported_names[name] == qualified_name:
            return Name(id=name, ctx=Load())
        alias = self.get_unused_name(name)
        node = self.load_names[module][name] = Name(id=alias, ctx=Load())
        self.imported_names[name] = qualified_name
        return node

    def insert_imports(self, node: Module | FunctionDef | AsyncFunctionDef) -> None:
        """Insert imports needed by injected code."""
        if not self.load_names:
            return
        for modulename, names in self.load_names.items():
            aliases = [alias(orig_name, new_name.id if orig_name != new_name.id else None) for orig_name, new_name in sorted(names.items())]
            node.body.insert(self.code_inject_index, ImportFrom(modulename, aliases, 0))

    def name_matches(self, expression: expr | Expr | None, *names: str) -> bool:
        if expression is None:
            return False
        path: list[str] = []
        top_expression = expression.value if isinstance(expression, Expr) else expression
        if isinstance(top_expression, Subscript):
            top_expression = top_expression.value
        elif isinstance(top_expression, Call):
            top_expression = top_expression.func
        while isinstance(top_expression, Attribute):
            path.insert(0, top_expression.attr)
            top_expression = top_expression.value
        if not isinstance(top_expression, Name):
            return False
        if top_expression.id in self.imported_names:
            translated = self.imported_names[top_expression.id]
        elif hasattr(builtins, top_expression.id):
            translated = 'builtins.' + top_expression.id
        else:
            translated = top_expression.id
        path.insert(0, translated)
        joined_path = '.'.join(path)
        if joined_path in names:
            return True
        elif self.parent:
            return self.parent.name_matches(expression, *names)
        else:
            return False

    def get_config_keywords(self) -> list[keyword]:
        if self.parent and isinstance(self.parent.node, ClassDef):
            overrides = self.parent.configuration_overrides.copy()
        else:
            overrides = {}
        overrides.update(self.configuration_overrides)
        return [keyword(key, value) for key, value in overrides.items()]