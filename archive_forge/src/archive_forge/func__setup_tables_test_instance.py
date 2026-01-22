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
@config.fixture(autouse=True, scope='function')
def _setup_tables_test_instance(self):
    self._setup_each_tables()
    self._setup_each_inserts()
    yield
    self._teardown_each_tables()