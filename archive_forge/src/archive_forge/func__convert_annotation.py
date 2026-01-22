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
def _convert_annotation(self, annotation: expr | None) -> expr | None:
    if annotation is None:
        return None
    new_annotation = cast(expr, AnnotationTransformer(self).visit(annotation))
    if isinstance(new_annotation, expr):
        new_annotation = ast.copy_location(new_annotation, annotation)
        names = {node.id for node in walk(new_annotation) if isinstance(node, Name)}
        self.names_used_in_annotations.update(names)
    return new_annotation