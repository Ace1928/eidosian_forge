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
@testing.combinations(('parameters',), ('multiparameters',), ('values',), ('omit',), argnames='insert_type')
def test_round_trip_none_as_sql_null(self, connection, insert_type):
    col = self.tables.data_table.c['nulldata']
    conn = connection
    if insert_type == 'parameters':
        stmt, params = (self.tables.data_table.insert(), {'name': 'r1', 'nulldata': None, 'data': None})
    elif insert_type == 'multiparameters':
        stmt, params = (self.tables.data_table.insert(), [{'name': 'r1', 'nulldata': None, 'data': None}])
    elif insert_type == 'values':
        stmt, params = (self.tables.data_table.insert().values(name='r1', nulldata=None, data=None), {})
    elif insert_type == 'omit':
        stmt, params = (self.tables.data_table.insert(), {'name': 'r1', 'data': None})
    else:
        assert False
    conn.execute(stmt, params)
    eq_(conn.scalar(select(self.tables.data_table.c.name).where(col.is_(null()))), 'r1')
    eq_(conn.scalar(select(col)), None)