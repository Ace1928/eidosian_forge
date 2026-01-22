from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
class SQLAsserter:

    def __init__(self):
        self.accumulated = []

    def _close(self):
        self._final = self.accumulated
        del self.accumulated

    def assert_(self, *rules):
        rule = EachOf(*rules)
        observed = list(self._final)
        while observed:
            statement = observed.pop(0)
            rule.process_statement(statement)
            if rule.is_consumed:
                break
            elif rule.errormessage:
                assert False, rule.errormessage
        if observed:
            assert False, 'Additional SQL statements remain:\n%s' % observed
        elif not rule.is_consumed:
            rule.no_more_statements()