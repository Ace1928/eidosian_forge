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
class UuidTest(_LiteralRoundTripFixture, fixtures.TablesTest):
    __backend__ = True
    datatype = Uuid

    @classmethod
    def define_tables(cls, metadata):
        Table('uuid_table', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('uuid_data', cls.datatype), Column('uuid_text_data', cls.datatype(as_uuid=False)), Column('uuid_data_nonnative', Uuid(native_uuid=False)), Column('uuid_text_data_nonnative', Uuid(as_uuid=False, native_uuid=False)))

    def test_uuid_round_trip(self, connection):
        data = uuid.uuid4()
        uuid_table = self.tables.uuid_table
        connection.execute(uuid_table.insert(), {'id': 1, 'uuid_data': data, 'uuid_data_nonnative': data})
        row = connection.execute(select(uuid_table.c.uuid_data, uuid_table.c.uuid_data_nonnative).where(uuid_table.c.uuid_data == data, uuid_table.c.uuid_data_nonnative == data)).first()
        eq_(row, (data, data))

    def test_uuid_text_round_trip(self, connection):
        data = str(uuid.uuid4())
        uuid_table = self.tables.uuid_table
        connection.execute(uuid_table.insert(), {'id': 1, 'uuid_text_data': data, 'uuid_text_data_nonnative': data})
        row = connection.execute(select(uuid_table.c.uuid_text_data, uuid_table.c.uuid_text_data_nonnative).where(uuid_table.c.uuid_text_data == data, uuid_table.c.uuid_text_data_nonnative == data)).first()
        eq_((row[0].lower(), row[1].lower()), (data, data))

    def test_literal_uuid(self, literal_round_trip):
        data = uuid.uuid4()
        literal_round_trip(self.datatype, [data], [data])

    def test_literal_text(self, literal_round_trip):
        data = str(uuid.uuid4())
        literal_round_trip(self.datatype(as_uuid=False), [data], [data], filter_=lambda x: x.lower())

    def test_literal_nonnative_uuid(self, literal_round_trip):
        data = uuid.uuid4()
        literal_round_trip(Uuid(native_uuid=False), [data], [data])

    def test_literal_nonnative_text(self, literal_round_trip):
        data = str(uuid.uuid4())
        literal_round_trip(Uuid(as_uuid=False, native_uuid=False), [data], [data], filter_=lambda x: x.lower())

    @testing.requires.insert_returning
    def test_uuid_returning(self, connection):
        data = uuid.uuid4()
        str_data = str(data)
        uuid_table = self.tables.uuid_table
        result = connection.execute(uuid_table.insert().returning(uuid_table.c.uuid_data, uuid_table.c.uuid_text_data, uuid_table.c.uuid_data_nonnative, uuid_table.c.uuid_text_data_nonnative), {'id': 1, 'uuid_data': data, 'uuid_text_data': str_data, 'uuid_data_nonnative': data, 'uuid_text_data_nonnative': str_data})
        row = result.first()
        eq_(row, (data, str_data, data, str_data))