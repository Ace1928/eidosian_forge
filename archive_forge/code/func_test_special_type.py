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
def test_special_type(self, metadata, connection, string_as_int):
    type_ = string_as_int
    t = Table('t', metadata, Column('x', type_))
    t.create(connection)
    connection.execute(t.insert(), [{'x': x} for x in [1, 2, 3]])
    result = {row[0] for row in connection.execute(t.select())}
    eq_(result, {1, 2, 3})
    result = {row[0] for row in connection.execute(t.select().where(t.c.x == 2))}
    eq_(result, {2})