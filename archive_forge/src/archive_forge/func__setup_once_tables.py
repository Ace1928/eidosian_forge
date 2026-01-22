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
@classmethod
def _setup_once_tables(cls):
    if cls.run_define_tables == 'once':
        cls.define_tables(cls._tables_metadata)
        if cls.run_create_tables == 'once':
            cls._tables_metadata.create_all(cls.bind)
        cls.tables.update(cls._tables_metadata.tables)
        cls.sequences.update(cls._tables_metadata._sequences)