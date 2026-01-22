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
def _teardown_each_tables(self):
    if self.run_define_tables == 'each':
        self.tables.clear()
        if self.run_create_tables == 'each':
            drop_all_tables_from_metadata(self._tables_metadata, self.bind)
        self._tables_metadata.clear()
    elif self.run_create_tables == 'each':
        drop_all_tables_from_metadata(self._tables_metadata, self.bind)
    savepoints = getattr(config.requirements, 'savepoints', False)
    if savepoints:
        savepoints = savepoints.enabled
    if self.run_define_tables != 'each' and self.run_create_tables != 'each' and (self.run_deletes == 'each'):
        with self.bind.begin() as conn:
            for table in reversed([t for t, fks in sort_tables_and_constraints(self._tables_metadata.tables.values()) if t is not None]):
                try:
                    if savepoints:
                        with conn.begin_nested():
                            conn.execute(table.delete())
                    else:
                        conn.execute(table.delete())
                except sa.exc.DBAPIError as ex:
                    print('Error emptying table %s: %r' % (table, ex), file=sys.stderr)