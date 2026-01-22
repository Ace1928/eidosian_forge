from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
def _received_statement(self, execute_observed):
    received_stmt, received_params = super()._received_statement(execute_observed)
    for real_stmt in execute_observed.statements:
        if self._compare_no_space(real_stmt.statement, received_stmt):
            break
    else:
        raise AssertionError("Can't locate compiled statement %r in list of statements actually invoked" % received_stmt)
    return (received_stmt, execute_observed.context.compiled_parameters)