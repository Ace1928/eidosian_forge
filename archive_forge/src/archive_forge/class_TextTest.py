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
class TextTest(_LiteralRoundTripFixture, fixtures.TablesTest):
    __requires__ = ('text_type',)
    __backend__ = True

    @property
    def supports_whereclause(self):
        return config.requirements.expressions_against_unbounded_text.enabled

    @classmethod
    def define_tables(cls, metadata):
        Table('text_table', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('text_data', Text))

    def test_text_roundtrip(self, connection):
        text_table = self.tables.text_table
        connection.execute(text_table.insert(), {'id': 1, 'text_data': 'some text'})
        row = connection.execute(select(text_table.c.text_data)).first()
        eq_(row, ('some text',))

    @testing.requires.empty_strings_text
    def test_text_empty_strings(self, connection):
        text_table = self.tables.text_table
        connection.execute(text_table.insert(), {'id': 1, 'text_data': ''})
        row = connection.execute(select(text_table.c.text_data)).first()
        eq_(row, ('',))

    def test_text_null_strings(self, connection):
        text_table = self.tables.text_table
        connection.execute(text_table.insert(), {'id': 1, 'text_data': None})
        row = connection.execute(select(text_table.c.text_data)).first()
        eq_(row, (None,))

    def test_literal(self, literal_round_trip):
        literal_round_trip(Text, ['some text'], ['some text'])

    @requirements.unicode_data_no_special_types
    def test_literal_non_ascii(self, literal_round_trip):
        literal_round_trip(Text, ['réve🐍 illé'], ['réve🐍 illé'])

    def test_literal_quoting(self, literal_round_trip):
        data = 'some \'text\' hey "hi there" that\'s text'
        literal_round_trip(Text, [data], [data])

    def test_literal_backslashes(self, literal_round_trip):
        data = 'backslash one \\ backslash two \\\\ end'
        literal_round_trip(Text, [data], [data])

    def test_literal_percentsigns(self, literal_round_trip):
        data = 'percent % signs %% percent'
        literal_round_trip(Text, [data], [data])