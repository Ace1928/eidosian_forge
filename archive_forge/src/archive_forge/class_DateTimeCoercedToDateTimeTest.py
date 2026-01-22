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
class DateTimeCoercedToDateTimeTest(_DateFixture, fixtures.TablesTest):
    """this particular suite is testing that datetime parameters get
    coerced to dates, which tends to be something DBAPIs do.

    """
    __requires__ = ('date', 'date_coerces_from_datetime')
    __backend__ = True
    datatype = Date
    data = datetime.datetime(2012, 10, 15, 12, 57, 18)
    compare = datetime.date(2012, 10, 15)

    @testing.requires.datetime_implicit_bound
    def test_select_direct(self, connection):
        result = connection.scalar(select(literal(self.data)))
        eq_(result, self.data)