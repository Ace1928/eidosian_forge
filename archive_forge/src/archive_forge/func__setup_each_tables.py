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
def _setup_each_tables(self):
    if self.run_define_tables == 'each':
        self.define_tables(self._tables_metadata)
        if self.run_create_tables == 'each':
            self._tables_metadata.create_all(self.bind)
        self.tables.update(self._tables_metadata.tables)
        self.sequences.update(self._tables_metadata._sequences)
    elif self.run_create_tables == 'each':
        self._tables_metadata.create_all(self.bind)