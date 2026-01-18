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
@testing.requires.insert_returning
def test_uuid_returning(self, connection):
    data = uuid.uuid4()
    str_data = str(data)
    uuid_table = self.tables.uuid_table
    result = connection.execute(uuid_table.insert().returning(uuid_table.c.uuid_data, uuid_table.c.uuid_text_data, uuid_table.c.uuid_data_nonnative, uuid_table.c.uuid_text_data_nonnative), {'id': 1, 'uuid_data': data, 'uuid_text_data': str_data, 'uuid_data_nonnative': data, 'uuid_text_data_nonnative': str_data})
    row = result.first()
    eq_(row, (data, str_data, data, str_data))