from __future__ import annotations
from collections import defaultdict
import contextlib
from copy import copy
from itertools import filterfalse
import re
import sys
import warnings
from . import assertsql
from . import config
from . import engines
from . import mock
from .exclusions import db_spec
from .util import fail
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import types as sqltypes
from .. import util
from ..engine import default
from ..engine import url
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import decorator
class AssertsExecutionResults:

    def assert_result(self, result, class_, *objects):
        result = list(result)
        print(repr(result))
        self.assert_list(result, class_, objects)

    def assert_list(self, result, class_, list_):
        self.assert_(len(result) == len(list_), 'result list is not the same size as test list, ' + 'for class ' + class_.__name__)
        for i in range(0, len(list_)):
            self.assert_row(class_, result[i], list_[i])

    def assert_row(self, class_, rowobj, desc):
        self.assert_(rowobj.__class__ is class_, 'item class is not ' + repr(class_))
        for key, value in desc.items():
            if isinstance(value, tuple):
                if isinstance(value[1], list):
                    self.assert_list(getattr(rowobj, key), value[0], value[1])
                else:
                    self.assert_row(value[0], getattr(rowobj, key), value[1])
            else:
                self.assert_(getattr(rowobj, key) == value, 'attribute %s value %s does not match %s' % (key, getattr(rowobj, key), value))

    def assert_unordered_result(self, result, cls, *expected):
        """As assert_result, but the order of objects is not considered.

        The algorithm is very expensive but not a big deal for the small
        numbers of rows that the test suite manipulates.
        """

        class immutabledict(dict):

            def __hash__(self):
                return id(self)
        found = util.IdentitySet(result)
        expected = {immutabledict(e) for e in expected}
        for wrong in filterfalse(lambda o: isinstance(o, cls), found):
            fail('Unexpected type "%s", expected "%s"' % (type(wrong).__name__, cls.__name__))
        if len(found) != len(expected):
            fail('Unexpected object count "%s", expected "%s"' % (len(found), len(expected)))
        NOVALUE = object()

        def _compare_item(obj, spec):
            for key, value in spec.items():
                if isinstance(value, tuple):
                    try:
                        self.assert_unordered_result(getattr(obj, key), value[0], *value[1])
                    except AssertionError:
                        return False
                elif getattr(obj, key, NOVALUE) != value:
                    return False
            return True
        for expected_item in expected:
            for found_item in found:
                if _compare_item(found_item, expected_item):
                    found.remove(found_item)
                    break
            else:
                fail('Expected %s instance with attributes %s not found.' % (cls.__name__, repr(expected_item)))
        return True

    def sql_execution_asserter(self, db=None):
        if db is None:
            from . import db as db
        return assertsql.assert_engine(db)

    def assert_sql_execution(self, db, callable_, *rules):
        with self.sql_execution_asserter(db) as asserter:
            result = callable_()
        asserter.assert_(*rules)
        return result

    def assert_sql(self, db, callable_, rules):
        newrules = []
        for rule in rules:
            if isinstance(rule, dict):
                newrule = assertsql.AllOf(*[assertsql.CompiledSQL(k, v) for k, v in rule.items()])
            else:
                newrule = assertsql.CompiledSQL(*rule)
            newrules.append(newrule)
        return self.assert_sql_execution(db, callable_, *newrules)

    def assert_sql_count(self, db, callable_, count):
        return self.assert_sql_execution(db, callable_, assertsql.CountStatements(count))

    @contextlib.contextmanager
    def assert_execution(self, db, *rules):
        with self.sql_execution_asserter(db) as asserter:
            yield
        asserter.assert_(*rules)

    def assert_statement_count(self, db, count):
        return self.assert_execution(db, assertsql.CountStatements(count))

    @contextlib.contextmanager
    def assert_statement_count_multi_db(self, dbs, counts):
        recs = [(self.sql_execution_asserter(db), db, count) for db, count in zip(dbs, counts)]
        asserters = []
        for ctx, db, count in recs:
            asserters.append(ctx.__enter__())
        try:
            yield
        finally:
            for asserter, (ctx, db, count) in zip(asserters, recs):
                ctx.__exit__(None, None, None)
                asserter.assert_(assertsql.CountStatements(count))