import datetime
import decimal
import json
import re
import uuid
from .. import config
from .. import engines
from .. import fixtures
from .. import mock
from ..assertions import eq_
from ..assertions import is_
from ..assertions import ne_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import and_
from ... import ARRAY
from ... import BigInteger
from ... import bindparam
from ... import Boolean
from ... import case
from ... import cast
from ... import Date
from ... import DateTime
from ... import Float
from ... import Integer
from ... import Interval
from ... import JSON
from ... import literal
from ... import literal_column
from ... import MetaData
from ... import null
from ... import Numeric
from ... import select
from ... import String
from ... import testing
from ... import Text
from ... import Time
from ... import TIMESTAMP
from ... import type_coerce
from ... import TypeDecorator
from ... import Unicode
from ... import UnicodeText
from ... import UUID
from ... import Uuid
from ...orm import declarative_base
from ...orm import Session
from ...sql import sqltypes
from ...sql.sqltypes import LargeBinary
from ...sql.sqltypes import PickleType
class _UnicodeFixture(_LiteralRoundTripFixture, fixtures.TestBase):
    __requires__ = ('unicode_data',)
    data = 'Alors vous imaginez ma 🐍 surprise, au lever du jour, quand une drôle de petite 🐍 voix m’a réveillé. Elle disait: « S’il vous plaît… dessine-moi 🐍 un mouton! »'

    @property
    def supports_whereclause(self):
        return config.requirements.expressions_against_unbounded_text.enabled

    @classmethod
    def define_tables(cls, metadata):
        Table('unicode_table', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('unicode_data', cls.datatype))

    def test_round_trip(self, connection):
        unicode_table = self.tables.unicode_table
        connection.execute(unicode_table.insert(), {'id': 1, 'unicode_data': self.data})
        row = connection.execute(select(unicode_table.c.unicode_data)).first()
        eq_(row, (self.data,))
        assert isinstance(row[0], str)

    def test_round_trip_executemany(self, connection):
        unicode_table = self.tables.unicode_table
        connection.execute(unicode_table.insert(), [{'id': i, 'unicode_data': self.data} for i in range(1, 4)])
        rows = connection.execute(select(unicode_table.c.unicode_data)).fetchall()
        eq_(rows, [(self.data,) for i in range(1, 4)])
        for row in rows:
            assert isinstance(row[0], str)

    def _test_null_strings(self, connection):
        unicode_table = self.tables.unicode_table
        connection.execute(unicode_table.insert(), {'id': 1, 'unicode_data': None})
        row = connection.execute(select(unicode_table.c.unicode_data)).first()
        eq_(row, (None,))

    def _test_empty_strings(self, connection):
        unicode_table = self.tables.unicode_table
        connection.execute(unicode_table.insert(), {'id': 1, 'unicode_data': ''})
        row = connection.execute(select(unicode_table.c.unicode_data)).first()
        eq_(row, ('',))

    def test_literal(self, literal_round_trip):
        literal_round_trip(self.datatype, [self.data], [self.data])

    def test_literal_non_ascii(self, literal_round_trip):
        literal_round_trip(self.datatype, ['réve🐍 illé'], ['réve🐍 illé'])