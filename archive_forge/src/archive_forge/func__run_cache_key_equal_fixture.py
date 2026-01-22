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
def _run_cache_key_equal_fixture(self, fixture, compare_values):
    case_a = fixture()
    case_b = fixture()
    for a, b in itertools.combinations_with_replacement(range(len(case_a)), 2):
        self._compare_equal(case_a[a], case_b[b], compare_values)