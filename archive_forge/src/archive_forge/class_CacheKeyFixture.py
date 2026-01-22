from __future__ import annotations
import itertools
import random
import re
import sys
import sqlalchemy as sa
from .base import TestBase
from .. import config
from .. import mock
from ..assertions import eq_
from ..assertions import ne_
from ..util import adict
from ..util import drop_all_tables_from_metadata
from ... import event
from ... import util
from ...schema import sort_tables_and_constraints
from ...sql import visitors
from ...sql.elements import ClauseElement
class CacheKeyFixture:

    def _compare_equal(self, a, b, compare_values):
        a_key = a._generate_cache_key()
        b_key = b._generate_cache_key()
        if a_key is None:
            assert a._annotations.get('nocache')
            assert b_key is None
        else:
            eq_(a_key.key, b_key.key)
            eq_(hash(a_key.key), hash(b_key.key))
            for a_param, b_param in zip(a_key.bindparams, b_key.bindparams):
                assert a_param.compare(b_param, compare_values=compare_values)
        return (a_key, b_key)

    def _run_cache_key_fixture(self, fixture, compare_values):
        case_a = fixture()
        case_b = fixture()
        for a, b in itertools.combinations_with_replacement(range(len(case_a)), 2):
            if a == b:
                a_key, b_key = self._compare_equal(case_a[a], case_b[b], compare_values)
                if a_key is None:
                    continue
            else:
                a_key = case_a[a]._generate_cache_key()
                b_key = case_b[b]._generate_cache_key()
                if a_key is None or b_key is None:
                    if a_key is None:
                        assert case_a[a]._annotations.get('nocache')
                    if b_key is None:
                        assert case_b[b]._annotations.get('nocache')
                    continue
                if a_key.key == b_key.key:
                    for a_param, b_param in zip(a_key.bindparams, b_key.bindparams):
                        if not a_param.compare(b_param, compare_values=compare_values):
                            break
                    else:
                        ne_(a_key.key, b_key.key)
                else:
                    ne_(a_key.key, b_key.key)
            if isinstance(case_a[a], ClauseElement) and isinstance(case_b[b], ClauseElement):
                assert_a_params = []
                assert_b_params = []
                for elem in visitors.iterate(case_a[a]):
                    if elem.__visit_name__ == 'bindparam':
                        assert_a_params.append(elem)
                for elem in visitors.iterate(case_b[b]):
                    if elem.__visit_name__ == 'bindparam':
                        assert_b_params.append(elem)
                eq_(sorted(a_key.bindparams, key=lambda b: b.key), sorted(util.unique_list(assert_a_params), key=lambda b: b.key))
                eq_(sorted(b_key.bindparams, key=lambda b: b.key), sorted(util.unique_list(assert_b_params), key=lambda b: b.key))

    def _run_cache_key_equal_fixture(self, fixture, compare_values):
        case_a = fixture()
        case_b = fixture()
        for a, b in itertools.combinations_with_replacement(range(len(case_a)), 2):
            self._compare_equal(case_a[a], case_b[b], compare_values)