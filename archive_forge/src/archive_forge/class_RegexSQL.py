from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
class RegexSQL(CompiledSQL):

    def __init__(self, regex, params=None, dialect='default', enable_returning=False):
        SQLMatchRule.__init__(self)
        self.regex = re.compile(regex)
        self.orig_regex = regex
        self.params = params
        self.dialect = dialect
        self.enable_returning = enable_returning

    def _failure_message(self, execute_observed, expected_params):
        return 'Testing for compiled statement ~%r partial params %s, received %%(received_statement)r with params %%(received_parameters)r' % (self.orig_regex.replace('%', '%%'), repr(expected_params).replace('%', '%%'))

    def _compare_sql(self, execute_observed, received_statement):
        return bool(self.regex.match(received_statement))