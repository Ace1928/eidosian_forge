from __future__ import annotations
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.spark import Spark
from sqlglot.tokens import TokenType
def _timestamp_diff(self: Databricks.Generator, expression: exp.DatetimeDiff | exp.TimestampDiff) -> str:
    return self.func('TIMESTAMPDIFF', expression.unit, expression.expression, expression.this)