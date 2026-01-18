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
@testing.combinations(('unicode', True), ('ascii', False), argnames='unicode_', id_='ia')
@testing.combinations(100, 1999, 3000, 4000, 5000, 9000, argnames='length')
def test_round_trip_pretty_large_data(self, connection, unicode_, length):
    if unicode_:
        data = 'réve🐍illé' * (length // 9 + 1)
        data = data[0:length // 2]
    else:
        data = 'abcdefg' * (length // 7 + 1)
        data = data[0:length]
    self._test_round_trip({'key1': data, 'key2': data}, connection)