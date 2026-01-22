from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
def _compare_sql(self, execute_observed, received_statement):
    stmt = self._dialect_adjusted_statement(execute_observed.context.dialect)
    return received_statement == stmt