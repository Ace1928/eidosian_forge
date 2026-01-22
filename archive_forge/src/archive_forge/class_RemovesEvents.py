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
class RemovesEvents:

    @util.memoized_property
    def _event_fns(self):
        return set()

    def event_listen(self, target, name, fn, **kw):
        self._event_fns.add((target, name, fn))
        event.listen(target, name, fn, **kw)

    @config.fixture(autouse=True, scope='function')
    def _remove_events(self):
        yield
        for key in self._event_fns:
            event.remove(*key)