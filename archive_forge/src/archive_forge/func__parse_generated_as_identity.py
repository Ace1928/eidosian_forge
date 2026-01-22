from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot.dialects.dialect import rename_func, unit_to_var
from sqlglot.dialects.hive import _build_with_ignore_nulls
from sqlglot.dialects.spark2 import Spark2, temporary_storage_provider
from sqlglot.helper import ensure_list, seq_get
from sqlglot.transforms import (
def _parse_generated_as_identity(self) -> exp.GeneratedAsIdentityColumnConstraint | exp.ComputedColumnConstraint | exp.GeneratedAsRowColumnConstraint:
    this = super()._parse_generated_as_identity()
    if this.expression:
        return self.expression(exp.ComputedColumnConstraint, this=this.expression)
    return this