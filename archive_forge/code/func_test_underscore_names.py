import random
from . import testing
from .. import config
from .. import fixtures
from .. import util
from ..assertions import eq_
from ..assertions import is_false
from ..assertions import is_true
from ..config import requirements
from ..schema import Table
from ... import CheckConstraint
from ... import Column
from ... import ForeignKeyConstraint
from ... import Index
from ... import inspect
from ... import Integer
from ... import schema
from ... import String
from ... import UniqueConstraint
@requirements.create_table
@util.provide_metadata
def test_underscore_names(self):
    table = self._underscore_fixture()
    table.create(config.db, checkfirst=False)
    self._simple_roundtrip(table)