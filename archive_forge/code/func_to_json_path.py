from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def to_json_path(self, path: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
    if isinstance(path, exp.Literal):
        path_text = path.name
        if path_text.startswith('/') or '[#' in path_text:
            return path
    return super().to_json_path(path)