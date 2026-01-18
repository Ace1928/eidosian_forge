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
@testing.requires.standalone_null_binds_whereclause
def test_null_bound_comparison(self):
    date_table = self.tables.date_table
    with config.db.begin() as conn:
        result = conn.execute(date_table.insert(), {'id': 1, 'date_data': self.data})
        id_ = result.inserted_primary_key[0]
        stmt = select(date_table.c.id).where(case((bindparam('foo', type_=self.datatype) != None, bindparam('foo', type_=self.datatype)), else_=date_table.c.date_data) == date_table.c.date_data)
        row = conn.execute(stmt, {'foo': None}).first()
        eq_(row[0], id_)