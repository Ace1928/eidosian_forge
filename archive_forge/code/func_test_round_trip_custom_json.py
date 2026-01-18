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
def test_round_trip_custom_json(self):
    data_table = self.tables.data_table
    data_element = {'key1': 'data1'}
    js = mock.Mock(side_effect=json.dumps)
    jd = mock.Mock(side_effect=json.loads)
    engine = engines.testing_engine(options=dict(json_serializer=js, json_deserializer=jd))
    data_table.create(engine, checkfirst=True)
    with engine.begin() as conn:
        conn.execute(data_table.insert(), {'name': 'row1', 'data': data_element})
        row = conn.execute(select(data_table.c.data)).first()
        eq_(row, (data_element,))
        eq_(js.mock_calls, [mock.call(data_element)])
        if testing.requires.json_deserializer_binary.enabled:
            eq_(jd.mock_calls, [mock.call(json.dumps(data_element).encode())])
        else:
            eq_(jd.mock_calls, [mock.call(json.dumps(data_element))])